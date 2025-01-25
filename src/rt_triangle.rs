//! Triangle types optimized for ray intersection performance.

use bytemuck::{Pod, Zeroable};
use glam::*;

use half::f16;

use crate::{aabb::Aabb, ray::Ray, triangle::Triangle, Boundable};

#[derive(Clone, Copy, Default, PartialEq)]
#[repr(C)]
/// A compressed 3D triangle optimized for GPU ray intersection performance.
pub struct RtCompressedTriangle {
    /// Base vertex
    pub v0: [f32; 3],
    /// Edges 1 & 2 encoded as IEEE 754 f16 `v1 - v0, v2 - v0`
    pub e1_e2: [u16; 6],
}

unsafe impl Pod for RtCompressedTriangle {}
unsafe impl Zeroable for RtCompressedTriangle {}

impl From<&Triangle> for RtCompressedTriangle {
    #[inline(always)]
    fn from(tri: &Triangle) -> Self {
        RtCompressedTriangle::new(tri.v0, tri.v1, tri.v2)
    }
}

impl RtCompressedTriangle {
    #[inline(always)]
    pub fn new(v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Self {
        let e1 = v1 - v0;
        let e2 = v2 - v0;

        Self {
            v0: [v0.x, v0.y, v0.z],
            e1_e2: [
                f16::from_f32(e1.x).to_bits(),
                f16::from_f32(e2.x).to_bits(),
                f16::from_f32(e1.y).to_bits(),
                f16::from_f32(e2.y).to_bits(),
                f16::from_f32(e1.z).to_bits(),
                f16::from_f32(e2.z).to_bits(),
            ],
        }
    }

    #[inline(always)]
    pub fn vertices(&self) -> [Vec3A; 3] {
        let (v0, e1, e2) = self.unpack();
        let v1 = v0 + e1;
        let v2 = v0 + e2;
        [v0, v1, v2]
    }

    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        Aabb::from_points(&self.vertices())
    }

    #[inline(always)]
    pub fn compute_normal(&self) -> Vec3A {
        let (_v0, e1, e2) = self.unpack();
        ((e1).cross(e2)).normalize_or_zero()
    }

    /// Find the distance (t) of the intersection of the `Ray` and this Triangle.
    /// Returns f32::INFINITY for miss.
    #[inline(always)]
    pub fn intersect(&self, ray: &Ray) -> f32 {
        // TODO not very water tight from the back side in some contexts (tris with edges at 0,0,0 show 1px gap)
        // Find out if this is typical of Möller
        // Based on Fast Minimum Storage Ray Triangle Intersection by T. Möller and B. Trumbore
        // https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html

        let (v0, e1, e2) = self.unpack();
        let ng = (-e1).cross(e2);

        let cull_backface = false;

        let c = v0 - ray.origin;
        let r = ray.direction.cross(c);
        let inv_det = 1.0 / ng.dot(ray.direction);

        let u = r.dot(e2) * inv_det;
        let v = r.dot(-e1) * inv_det;
        let w = 1.0 - u - v;

        // Original:
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
            let t = ng.dot(c) * inv_det;
            if t >= ray.tmin && t <= ray.tmax {
                return t;
            }
        }

        f32::INFINITY
    }

    pub fn unpack(&self) -> (Vec3A, Vec3A, Vec3A) {
        let v0: Vec3A = self.v0.into();
        let e1x = f16::from_bits(self.e1_e2[0]).to_f32();
        let e2x = f16::from_bits(self.e1_e2[1]).to_f32();
        let e1y = f16::from_bits(self.e1_e2[2]).to_f32();
        let e2y = f16::from_bits(self.e1_e2[3]).to_f32();
        let e1z = f16::from_bits(self.e1_e2[4]).to_f32();
        let e2z = f16::from_bits(self.e1_e2[5]).to_f32();
        let e1 = Vec3A::new(e1x, e1y, e1z);
        let e2 = Vec3A::new(e2x, e2y, e2z);
        (v0, e1, e2)
    }

    #[inline(always)]
    pub fn compute_barycentric(&self, ray: &Ray) -> Vec2 {
        let (v0, e1, e2) = self.unpack();
        let ng = (-e1).cross(e2);
        let r = ray.direction.cross(v0 - ray.origin);
        vec2(r.dot(e2), r.dot(-e1)) / ng.dot(ray.direction)
    }
}

impl Boundable for RtCompressedTriangle {
    fn aabb(&self) -> Aabb {
        self.aabb()
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
/// A 3D triangle optimized for CPU ray intersection performance.
pub struct RtTriangle {
    /// Base vertex
    pub v0: Vec3A,
    /// Edge 1 `v0 - v1`
    pub e1: Vec3A,
    /// Edge 2 `v2 - v0`
    pub e2: Vec3A,
    /// Geometric normal `e1.cross(e2)`.
    /// Optimized for intersection.
    /// Needs to be inverted for typical normal.
    pub ng: Vec3A,
}

impl From<&Triangle> for RtTriangle {
    #[inline(always)]
    fn from(tri: &Triangle) -> Self {
        RtTriangle::new(tri.v0, tri.v1, tri.v2)
    }
}

// Uses layout from https://github.com/madmann91/bvh/blob/master/src/bvh/v2/tri.h#L36
// to optimize for intersection. On the CPU this is a bit faster than e1 = v1 - v0; e2 = v2 - v0;
impl RtTriangle {
    #[inline(always)]
    pub fn new(v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Self {
        let e1 = v0 - v1;
        let e2 = v2 - v0;
        Self {
            v0,
            e1,
            e2,
            ng: e1.cross(e2),
        }
    }

    #[inline(always)]
    fn vertices(&self) -> [Vec3A; 3] {
        [self.v0, self.v0 - self.e1, self.v0 + self.e2]
    }

    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        Aabb::from_points(&self.vertices())
    }

    #[inline(always)]
    pub fn compute_normal(&self) -> Vec3A {
        -self.ng.normalize_or_zero()
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

        let c = self.v0 - ray.origin;
        let r = ray.direction.cross(c);
        let inv_det = 1.0 / self.ng.dot(ray.direction);

        let u = r.dot(self.e2) * inv_det;
        let v = r.dot(self.e1) * inv_det;
        let w = 1.0 - u - v;

        // Original:
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
            let t = self.ng.dot(c) * inv_det;
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
        // Not watertight from the front side? Looks similar to what intersect() above looks like from the back side.

        // This uses the orientation from https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html

        let cull_backface = false;

        // Calculate denominator
        let o = ray.origin;
        let d = ray.direction;
        let c = self.v0 - o;
        let r = c.cross(d);
        let den = (-self.ng).dot(d);
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
        let u = f32::from_bits(r.dot(self.e2).to_bits() ^ sgn_den);
        let v = f32::from_bits(r.dot(self.e1).to_bits() ^ sgn_den);
        // TODO simd uv?

        // Perform backface culling
        // Original:
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
        let t = f32::from_bits((-self.ng).dot(c).to_bits() ^ sgn_den);

        if abs_den * ray.tmin < t && t <= abs_den * ray.tmax {
            return t;
        }

        f32::INFINITY
    }

    #[inline(always)]
    pub fn compute_barycentric(&self, ray: &Ray) -> Vec2 {
        let r = ray.direction.cross(self.v0 - ray.origin);
        vec2(r.dot(self.e2), r.dot(self.e1)) / self.ng.dot(ray.direction)
    }
}

impl Boundable for RtTriangle {
    fn aabb(&self) -> Aabb {
        self.aabb()
    }
}
