//! Meshes, generators, sampling functions, etc.. for basic testing & examples.

pub mod sampling {
    use std::f32::consts::TAU;

    use glam::*;

    #[inline(always)]
    pub fn uhash(x: u32) -> u32 {
        // from https://nullprogram.com/blog/2018/07/31/
        let mut x = x ^ (x >> 16);
        x = x.overflowing_mul(0x7feb352d).0;
        x = x ^ (x >> 15);
        x = x.overflowing_mul(0x846ca68b).0;
        x = x ^ (x >> 16);
        x
    }

    pub fn hash_f32_vec(v: &[f32]) -> u32 {
        v.iter()
            .map(|&f| uhash(f.to_bits()))
            .fold(0, |acc, h| acc ^ h)
    }

    pub fn hash_vec3a_vec(v: &[Vec3A]) -> u32 {
        v.iter()
            .flat_map(|v| [v.x, v.y, v.z])
            .map(|f| uhash(f.to_bits()))
            .fold(0, |acc, h| acc ^ h)
    }

    #[inline(always)]
    pub fn uhash2(a: u32, b: u32) -> u32 {
        uhash((a.overflowing_mul(1597334673).0) ^ (b.overflowing_mul(3812015801).0))
    }

    #[inline(always)]
    pub fn unormf(n: u32) -> f32 {
        n as f32 * (1.0 / 0xffffffffu32 as f32)
    }

    #[inline(always)]
    pub fn hash_noise(coord: UVec2, frame: u32) -> f32 {
        let urnd = uhash2(coord.x, (coord.y << 11) + frame);
        unormf(urnd)
    }

    // https://jcgt.org/published/0006/01/01/paper.pdf
    #[inline(always)]
    pub fn build_orthonormal_basis(n: Vec3A) -> Mat3 {
        let sign = n.z.signum();
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;

        mat3(
            vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
            vec3(b, sign + n.y * n.y * a, -n.y),
            n.into(),
        )
    }

    #[inline(always)]
    pub fn cosine_sample_hemisphere(urand: Vec2) -> Vec3A {
        let r = urand.x.sqrt();
        let theta = urand.y * TAU;
        vec3a(
            r * theta.cos(),
            r * theta.sin(),
            0.0f32.max(1.0 - urand.x).sqrt(),
        )
    }

    #[inline(always)]
    pub fn uniform_sample_sphere(urand: Vec2) -> Vec3A {
        let z = 1.0 - 2.0 * urand.x;
        let r = (1.0 - z * z).sqrt();
        let theta = urand.y * TAU;
        vec3a(r * theta.cos(), r * theta.sin(), z)
    }

    #[inline(always)]
    pub fn uniform_sample_cone(urand: Vec2, cos_theta_max: f32) -> Vec3A {
        let cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
        let sin_theta = (1.0 - cos_theta * cos_theta).clamp(0.0, 1.0).sqrt();
        let phi: f32 = urand.y * TAU;
        vec3a(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
    }

    #[inline(always)]
    pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[inline(always)]
    fn cubic(v0: f32, v1: f32, v2: f32, v3: f32, x: f32) -> f32 {
        let p = (v3 - v2) - (v0 - v1);
        let q = (v0 - v1) - p;
        let r = v2 - v0;
        let s = v1;
        p * x.powi(3) + q * x.powi(2) + r * x + s
    }

    #[inline(always)]
    pub fn bicubic_noise(coord: Vec2, seed: u32) -> f32 {
        let ix = coord.x.floor() as u32;
        let iy = coord.y.floor() as u32;
        let fx = coord.x - ix as f32;
        let fy = coord.y - iy as f32;
        fn cubic_col(ix: u32, iy: u32, j: u32, seed: u32, fx: f32) -> f32 {
            cubic(
                hash_noise(uvec2(ix, iy + j), seed),
                hash_noise(uvec2(ix + 1, iy + j), seed),
                hash_noise(uvec2(ix + 2, iy + j), seed),
                hash_noise(uvec2(ix + 3, iy + j), seed),
                fx,
            )
        }
        cubic(
            cubic_col(ix, iy, 0, seed, fx),
            cubic_col(ix, iy, 1, seed, fx),
            cubic_col(ix, iy, 2, seed, fx),
            cubic_col(ix, iy, 3, seed, fx),
            fy,
        )
    }

    // By Tomasz Stachowiak
    pub fn somewhat_boring_display_transform(col: Vec3A) -> Vec3A {
        fn rgb_to_ycbcr(col: Vec3A) -> Vec3A {
            Mat3A {
                x_axis: vec3a(0.2126, -0.1146, 0.5),
                y_axis: vec3a(0.7152, -0.3854, -0.4542),
                z_axis: vec3a(0.0722, 0.5, -0.0458),
            } * col
        }

        fn tonemap_curve(v: f32) -> f32 {
            1.0 - (-v).exp()
        }

        fn tonemap_curve3(v: Vec3A) -> Vec3A {
            1.0 - (-v).exp()
        }

        fn tonemapping_luminance(col: Vec3A) -> f32 {
            col.dot(vec3a(0.2126, 0.7152, 0.0722))
        }

        let mut col = col;
        let ycbcr = rgb_to_ycbcr(col);

        let bt = tonemap_curve(ycbcr.yz().length() * 2.4);
        let mut desat = (bt - 0.7) * 0.8;
        desat *= desat;

        let desat_col = col.lerp(ycbcr.xxx(), desat);

        let tm_luma = tonemap_curve(ycbcr.x);
        let tm0 = col * tm_luma / tonemapping_luminance(col).max(1e-5);
        let final_mult = 0.97;
        let tm1 = tonemap_curve3(desat_col);

        col = tm0.lerp(tm1, bt * bt);

        col * final_mult
    }
}

pub mod geometry {
    use crate::{Triangle, test_util::sampling::bicubic_noise};
    use glam::*;

    #[inline(always)]
    const fn vec(a: f32, b: f32, c: f32) -> Vec3A {
        Vec3A::new(a, b, c)
    }
    #[inline(always)]
    const fn tri(v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Triangle {
        Triangle { v0, v1, v2 }
    }

    /// Cube triangle mesh with side length of 2 centered at 0,0,0
    pub const CUBE: [Triangle; 12] = [
        tri(vec(-1., 1., -1.), vec(1., 1., 1.), vec(1., 1., -1.)),
        tri(vec(1., 1., 1.), vec(-1., -1., 1.), vec(1., -1., 1.)),
        tri(vec(-1., 1., 1.), vec(-1., -1., -1.), vec(-1., -1., 1.)),
        tri(vec(1., -1., -1.), vec(-1., -1., 1.), vec(-1., -1., -1.)),
        tri(vec(1., 1., -1.), vec(1., -1., 1.), vec(1., -1., -1.)),
        tri(vec(-1., 1., -1.), vec(1., -1., -1.), vec(-1., -1., -1.)),
        tri(vec(-1., 1., -1.), vec(-1., 1., 1.), vec(1., 1., 1.)),
        tri(vec(1., 1., 1.), vec(-1., 1., 1.), vec(-1., -1., 1.)),
        tri(vec(-1., 1., 1.), vec(-1., 1., -1.), vec(-1., -1., -1.)),
        tri(vec(1., -1., -1.), vec(1., -1., 1.), vec(-1., -1., 1.)),
        tri(vec(1., 1., -1.), vec(1., 1., 1.), vec(1., -1., 1.)),
        tri(vec(-1., 1., -1.), vec(1., 1., -1.), vec(1., -1., -1.)),
    ];

    /// Plane triangle mesh with side length of 2 centered at 0,0,0
    pub const PLANE: [Triangle; 2] = [
        tri(vec(1., 0., 1.), vec(-1., 0., -1.), vec(-1., 0., 1.)),
        tri(vec(1., 0., 1.), vec(1., 0., -1.), vec(-1., 0., -1.)),
    ];

    /// Generate icosphere mesh with radius of 2
    pub fn icosphere(subdivisions: u32) -> Vec<Triangle> {
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0; // golden ratio
        let (a, b, c, d, e) = (1.0, -1.0, 0.0, phi, -phi);

        #[rustfmt::skip]
        let mut p = [vec(b,d,c),vec(a,d,c),vec(b,e,c),vec(a,e,c),vec(c,b,d),vec(c,a,d),vec(c,b,e),vec(c,a,e),vec(d,c,b),vec(d,c,a),vec(e,c,b),vec(e,c,a)];
        p.iter_mut().for_each(|v| *v = v.normalize());

        let mut tris = vec![
            tri(p[0], p[11], p[5]),
            tri(p[0], p[5], p[1]),
            tri(p[0], p[1], p[7]),
            tri(p[0], p[7], p[10]),
            tri(p[0], p[10], p[11]),
            tri(p[1], p[5], p[9]),
            tri(p[5], p[11], p[4]),
            tri(p[11], p[10], p[2]),
            tri(p[10], p[7], p[6]),
            tri(p[7], p[1], p[8]),
            tri(p[3], p[9], p[4]),
            tri(p[3], p[4], p[2]),
            tri(p[3], p[2], p[6]),
            tri(p[3], p[6], p[8]),
            tri(p[3], p[8], p[9]),
            tri(p[4], p[9], p[5]),
            tri(p[2], p[4], p[11]),
            tri(p[6], p[2], p[10]),
            tri(p[8], p[6], p[7]),
            tri(p[9], p[8], p[1]),
        ];

        (0..subdivisions).for_each(|_| {
            let mut new_tris = Vec::new();
            tris.iter().for_each(|t| {
                let mid01 = ((t.v0 + t.v1) * 0.5).normalize();
                let mid12 = ((t.v1 + t.v2) * 0.5).normalize();
                let mid20 = ((t.v2 + t.v0) * 0.5).normalize();
                new_tris.push(tri(t.v0, mid01, mid20));
                new_tris.push(tri(t.v1, mid12, mid01));
                new_tris.push(tri(t.v2, mid20, mid12));
                new_tris.push(tri(mid01, mid12, mid20));
            });
            tris = new_tris;
        });

        tris
    }

    /// Convert height map to triangles with 2x2x2 size given -1.0..=1.0 output from height_map: F
    pub fn height_to_triangles<F>(
        height_map: F,
        x_resolution: usize,
        z_resolution: usize,
    ) -> Vec<Triangle>
    where
        F: Fn(usize, usize) -> f32,
    {
        let mut triangles = Vec::new();

        // Iterate over each cell in the grid
        for z in 0..z_resolution {
            for x in 0..x_resolution {
                // Calculate normalized positions
                let fx = (x as f32 / x_resolution as f32) * 2.0 - 1.0;
                let fz = (z as f32 / z_resolution as f32) * 2.0 - 1.0;
                let fx2 = ((x + 1) as f32 / x_resolution as f32) * 2.0 - 1.0;
                let fz2 = ((z + 1) as f32 / z_resolution as f32) * 2.0 - 1.0;

                // Create vertices for each corner of the cell
                let v00 = vec(fx, height_map(x, z), fz);
                let v10 = vec(fx2, height_map(x + 1, z), fz);
                let v01 = vec(fx, height_map(x, z + 1), fz2);
                let v11 = vec(fx2, height_map(x + 1, z + 1), fz2);

                // Create two triangles for this cell
                triangles.push(tri(v00, v01, v10));
                triangles.push(tri(v10, v01, v11));
            }
        }

        triangles
    }

    /// terrain_res 1024 or greater recommended
    pub fn demoscene(terrain_res: usize, seed: u32) -> Vec<Triangle> {
        let height_map = |x: usize, y: usize| -> f32 {
            let coord = vec2(x as f32, y as f32) / terrain_res as f32;
            let (mut cs, mut ns) = (1.579, 0.579);
            (1..17)
                .map(|i| {
                    (cs, ns) = (cs * 1.579, ns * -0.579);
                    bicubic_noise(coord * cs, seed + i) * ns
                })
                .sum::<f32>()
                * (1.0 - coord.y).powf(0.579)
                + (1.0 - coord.y).powf(1.579) * 0.579
        };
        height_to_triangles(height_map, terrain_res, terrain_res)
    }
}
