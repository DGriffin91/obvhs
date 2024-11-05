//! Triangle representation in 3D space.

use bytemuck::{Pod, Zeroable};
use glam::{vec2, Mat4, Vec2, Vec3A};

use crate::{aabb::Aabb, ray::Ray, Boundable, Transformable};

#[derive(Clone, Copy, Default, Debug)]
pub struct Triangle {
    pub v0: Vec3A,
    pub v1: Vec3A,
    pub v2: Vec3A,
}

unsafe impl Pod for Triangle {}
unsafe impl Zeroable for Triangle {}

impl Triangle {
    /// Compute the normal of the triangle geometry.
    #[inline(always)]
    pub fn compute_normal(&self) -> Vec3A {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2).normalize_or_zero()
    }

    /// Compute the bounding box of the triangle.
    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        Aabb::from_points(&[self.v0, self.v1, self.v2])
    }

    /// Find the distance (t) of the intersection of the `Ray` and this Triangle.
    /// Returns f32::INFINITY for miss.
    #[inline(always)]
    pub fn intersect(&self, ray: &Ray) -> f32 {
        // TODO not very water tight from the back side in some contexts (tris with edges at 0,0,0 show 1px gap)
        // Find out if this is typical of Möller
        // Based on Fast Minimum Storage Ray Triangle Intersection by T. Möller and B. Trumbore
        // https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html
        let cull_backface = false;
        let e1 = self.v0 - self.v1;
        let e2 = self.v2 - self.v0;
        let n = e1.cross(e2);

        let c = self.v0 - ray.origin;
        let r = ray.direction.cross(c);
        let inv_det = 1.0 / n.dot(ray.direction);

        let u = r.dot(e2) * inv_det;
        let v = r.dot(e1) * inv_det;
        let w = 1.0 - u - v;

        //let hit = u >= 0.0 && v >= 0.0 && w >= 0.0;
        //let valid = if cull_backface {
        //    inv_det > 0.0 && hit
        //} else {
        //    inv_det != 0.0 && hit
        //};

        // Note: differs in that if v == -0.0, for example will cause valid to be false
        let hit = u.to_bits() | v.to_bits() | w.to_bits();
        let valid = if cull_backface {
            (inv_det.to_bits() | hit) & 0x8000_0000 == 0
        } else {
            inv_det != 0.0 && hit & 0x8000_0000 == 0
        };

        if valid {
            let t = n.dot(c) * inv_det;
            if t >= ray.tmin && t <= ray.tmax {
                return t;
            }
        }

        f32::INFINITY
    }

    // https://github.com/RenderKit/embree/blob/0c236df6f31a8e9c8a48803dada333e9ea0029a6/kernels/geometry/triangle_intersector_moeller.h#L9
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    pub fn intersect_embree(&self, ray: &Ray) -> f32 {
        // Not watertight from the front side? Looks similar to what above looks like from the back side.

        // This uses the orientation from https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html

        let cull_backface = false;
        let v0 = self.v0;
        let e1 = self.v0 - self.v1;
        let e2 = self.v2 - self.v0;
        let ng = e1.cross(e2);

        // Calculate denominator
        let o = ray.origin;
        let d = ray.direction;
        let c = v0 - o;
        let r = c.cross(d);
        let den = (-ng).dot(d);
        let abs_den = den.abs();

        fn signmsk(x: f32) -> f32 {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            unsafe {
                let mask = _mm_set1_ps(-0.0);
                let x_vec = _mm_set_ss(x);
                let sign_bit = _mm_and_ps(x_vec, mask);
                _mm_cvtss_f32(sign_bit)
                //_mm_cvtss_f32(_mm_and_ps(
                //    _mm_set_ss(x),
                //    _mm_castsi128_ps(_mm_set1_epi32(-2147483648i32)),
                //))
            }
        }

        let sgn_den = signmsk(den).to_bits();

        // Perform edge tests
        let u = f32::from_bits(r.dot(e2).to_bits() ^ sgn_den);
        let v = f32::from_bits(r.dot(e1).to_bits() ^ sgn_den);
        // TODO simd uv?

        // Perform backface culling
        // OG
        //let valid = if cull_backface {
        //    den < 0.0 && u >= 0.0 && v >= 0.0 && u + v <= abs_den
        //} else {
        //    den != 0.0 && u >= 0.0 && v >= 0.0 && u + v <= abs_den
        //};

        let w = abs_den - u - v;
        let valid = if cull_backface {
            ((-den).to_bits() | u.to_bits() | v.to_bits() | (abs_den - u - v).to_bits())
                & 0x8000_0000
                == 0
        } else {
            den != 0.0 && ((u.to_bits() | v.to_bits() | w.to_bits()) & 0x8000_0000) == 0
        };

        if !valid {
            return f32::INFINITY;
        }

        // Perform depth test
        let t = f32::from_bits((-ng).dot(c).to_bits() ^ sgn_den);

        if abs_den * ray.tmin < t && t <= abs_den * ray.tmax {
            return t;
        }

        f32::INFINITY
    }

    #[inline(always)]
    pub fn compute_barycentric(&self, ray: &Ray) -> Vec2 {
        let e1 = self.v0 - self.v1;
        let e2 = self.v2 - self.v0;
        let ng = e1.cross(e2).normalize_or_zero();
        let r = ray.direction.cross(self.v0 - ray.origin);
        vec2(r.dot(e2), r.dot(e1)) / ng.dot(ray.direction)
    }
}

impl Boundable for Triangle {
    fn aabb(&self) -> Aabb {
        self.aabb()
    }
}

impl Transformable for &mut Triangle {
    fn transform(&mut self, matrix: &Mat4) {
        self.v0 = matrix.transform_point3a(self.v0);
        self.v1 = matrix.transform_point3a(self.v1);
        self.v2 = matrix.transform_point3a(self.v2);
    }
}

impl<T> Transformable for T
where
    T: AsMut<[Triangle]>,
{
    fn transform(&mut self, matrix: &Mat4) {
        self.as_mut().iter_mut().for_each(|mut triangle| {
            triangle.transform(matrix);
        });
    }
}
