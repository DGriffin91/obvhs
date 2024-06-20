// For fun, not pbr
// Run with `--release --features parallel` unless you like waiting around for a very long time.
use glam::*;
use image::{ImageBuffer, Rgba};
use obvhs::{
    cwbvh::builder::build_cwbvh_from_tris,
    ray::{Ray, RayHit},
    rt_triangle::RtTriangle,
    test_util::{
        geometry::demoscene,
        sampling::{
            build_orthonormal_basis, cosine_sample_hemisphere, hash_noise,
            somewhat_boring_display_transform, uniform_sample_cone, uniform_sample_sphere,
        },
    },
    timeit, BvhBuildParams,
};
use std::io::Write;
pub const SUN_ANGULAR_DIAMETER: f32 = 0.00933;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sky::Sky;

fn main() {
    let total_aa_samples = 64;
    let resolution = 2560;
    let seed = 57;

    timeit!["generate height map",
    let tris = demoscene(resolution as usize, seed * 10);
    ];
    println!("{} triangles, {} AA samples", tris.len(), total_aa_samples);
    timeit!["generate bvh",
    let bvh = build_cwbvh_from_tris(&tris, BvhBuildParams::very_fast_build(), &mut 0.0);
    ];

    let bvh_tris = bvh
        .primitive_indices
        .iter()
        .map(|i| (&tris[*i as usize]).into())
        .collect::<Vec<RtTriangle>>();

    let intersection_fn = |ray: &Ray, id: usize| bvh_tris[id].intersect(ray);

    // Setup render target and camera
    let width = resolution;
    let height = ((resolution as f32) * 0.3711) as u32;
    let target_size = Vec2::new(width as f32, height as f32);
    let fov = 17.0f32;
    let eye = vec3a(0.0, 0.0, 1.35);
    let look_at = eye + vec3a(0.0, 0.16, -1.0);
    let sun_direction = vec3a(0.35, -0.1, 0.19).normalize();
    let sky = Sky::red_sunset(-sun_direction);
    let sky_bg = Sky::red_sunset(-vec3a(0.35, -0.1, 0.5).normalize()); // To extend the sun glow a bit in the BG
    let nee = 1.0 - SUN_ANGULAR_DIAMETER.cos();
    let material_color = vec3a(0.61, 0.59, 0.52).powf(2.2);
    let exposure = -3.6;

    // Compute camera projection & view matrices
    let aspect_ratio = target_size.x / target_size.y;
    let proj_inv =
        Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect_ratio, 0.01).inverse();
    let view = Mat4::look_at_rh(eye.into(), look_at.into(), Vec3::Y);
    let view_inv = view.inverse();

    let mut fragments = vec![Vec3A::ZERO; (width * height) as usize];

    println!("|{}|", " ".repeat(total_aa_samples as usize));
    print!(" ");
    timeit![
        "render",
        for aa_sample in 0..total_aa_samples {
            print!("."); // Print progress
            std::io::stdout().flush().unwrap();
            let new_fragments: Vec<Vec3A>;
            #[cfg(feature = "parallel")]
            let iter = (0..width * height).into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = (0..width * height).into_iter();
            new_fragments = iter
                .map(|i| {
                    let frag_coord = uvec2(i as u32 % width, i as u32 / width);
                    let misc_grain_noise = hash_noise(frag_coord, aa_sample + 12345);
                    let aa = vec2(
                        hash_noise(frag_coord, aa_sample),
                        hash_noise(frag_coord, aa_sample + 512),
                    ) * 0.5
                        - 0.25;
                    let mut screen_uv = (frag_coord.as_vec2() + aa) / target_size;
                    screen_uv.y = 1.0 - screen_uv.y;
                    let ndc = screen_uv * 2.0 - Vec2::ONE;
                    let clip_pos = vec4(ndc.x, ndc.y, 1.0, 1.0);

                    let mut vs = proj_inv * clip_pos;
                    vs /= vs.w;
                    let direction = (Vec3A::from((view_inv * vs).xyz()) - eye).normalize();

                    let fuzz = vec3a(
                        hash_noise(frag_coord, aa_sample),
                        hash_noise(frag_coord, aa_sample + 512),
                        hash_noise(frag_coord, aa_sample + 1024),
                    );
                    let fuzzy_cube_of_sensor = eye + (fuzz * 2.0 - 1.0) * 0.002;

                    let focal_distance = 2.4;
                    let focal_point = eye + direction * focal_distance;
                    let cam_dir = (focal_point - fuzzy_cube_of_sensor).normalize_or_zero();
                    let ray = Ray::new_inf(fuzzy_cube_of_sensor, cam_dir);

                    let mut color = Vec3A::ZERO;

                    let fog_dir = uniform_sample_sphere(vec2(
                        hash_noise(frag_coord, aa_sample + 2048),
                        hash_noise(frag_coord, aa_sample + 3840),
                    ));
                    let mut hit = RayHit::none();
                    let fogc = sky.render(fog_dir).min(Vec3A::splat(100.0));
                    let skyc = sky.render(ray.direction);
                    let sunc = sky.render(-sun_direction);
                    let mut state = bvh.new_traversal(ray);
                    while bvh.traverse_dynamic(&mut state, &mut hit, intersection_fn) {}
                    if hit.t < f32::MAX {
                        let mut normal = bvh_tris[hit.primitive_id as usize].compute_normal();
                        normal *= normal.dot(-ray.direction).signum(); // Double sided

                        let hit_p = ray.origin + ray.direction * hit.t - ray.direction * 0.01;

                        let tangent_to_world = build_orthonormal_basis(normal);
                        let mut ao_ray_dir = cosine_sample_hemisphere(vec2(
                            hash_noise(frag_coord, aa_sample),
                            hash_noise(frag_coord, aa_sample + 1024),
                        ));
                        ao_ray_dir = (tangent_to_world * ao_ray_dir).normalize();

                        let diff_ray = Ray::new_inf(hit_p, ao_ray_dir);
                        let mut diff_hit = RayHit::none();
                        state.reinit(diff_ray);
                        while bvh.traverse_dynamic(&mut state, &mut diff_hit, intersection_fn) {}
                        if diff_hit.t < f32::MAX {
                            let mut diff_hit_normal =
                                bvh_tris[diff_hit.primitive_id as usize].compute_normal();
                            diff_hit_normal *= diff_hit_normal.dot(-ray.direction).signum(); // Double sided

                            // Silly 1st bounce sun shadow ray
                            let ao_hit_p = hit_p + diff_ray.direction * diff_hit.t - diff_ray.direction * 0.01;
                            let sun_ray = Ray::new_inf(ao_hit_p, -sun_direction);
                            let mut sun_hit = RayHit::none();
                            // anyhit

                            state.reinit(sun_ray);
                            if !bvh.traverse_dynamic(&mut state, &mut sun_hit, intersection_fn) {
                                // xD
                                color += material_color * material_color * nee * sunc * 4.0;
                            }
                        } else {
                            let fresnel = (1.0 - normal.dot(-cam_dir)).powf(8.0).max(0.0);
                            let skyc = sky
                                .render(diff_ray.direction)
                                // Sun results in fireflies. Clamp to avoid randomly sampling super high values.
                                .min(Vec3A::splat(100.0));
                            color += material_color * (fresnel * skyc * 0.5 + skyc);
                        }

                        // Sun shadow ray
                        let sun_rnd = vec2(
                            hash_noise(frag_coord, aa_sample + 10000),
                            hash_noise(frag_coord, aa_sample + 20000),
                        );
                        let sun_basis = build_orthonormal_basis(sun_direction);
                        let sun_dir = (sun_basis
                            * uniform_sample_cone(sun_rnd, (SUN_ANGULAR_DIAMETER * 0.5).cos()))
                        .normalize_or_zero();

                        let mut sun_hit = RayHit::none();
                        let sun_ray = Ray::new_inf(hit_p, -sun_dir);

                        state.reinit(sun_ray); // anyhit
                        if !bvh.traverse_dynamic(&mut state, &mut sun_hit, intersection_fn) {
                            color += material_color
                                * nee
                                * normal.dot(-sun_dir).max(0.00001)
                                * sunc
                                * 10.0
                                * misc_grain_noise;
                        }

                        // Fog shadow ray
                        let fog_t = hit.t * hash_noise(frag_coord, aa_sample + 54321);
                        let fog_p = ray.origin + ray.direction * fog_t;
                        let sun_ray = Ray::new_inf(fog_p, -sun_direction);
                        let mut sun_hit = RayHit::none();

                        state.reinit(sun_ray); // anyhit
                        if !bvh.traverse_dynamic(&mut state, &mut sun_hit, intersection_fn) {
                            color += nee * sunc * fog_t * 0.2;
                        }
                        state.reinit(Ray::new_inf(fog_p, fog_dir)); // anyhit
                        if !bvh.traverse_dynamic(&mut state, &mut RayHit::none(), intersection_fn) {
                            color += fog_t * 0.2 * fogc;
                        }
                    } else {
                        let sky_bgc = sky_bg.render(ray.direction) * 0.4 + skyc * 0.6;
                        color += sky_bgc * 0.4 + sky_bgc * misc_grain_noise * 0.6;
                        color += 0.2 * fogc;
                    }
                    color
                })
                .collect::<Vec<_>>();
            new_fragments
                .iter()
                .zip(fragments.iter_mut())
                .for_each(|(new, col)| *col += *new);
        }
        println!("");
    ];

    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let pixels = img.as_mut();
    pixels.chunks_mut(4).enumerate().for_each(|(i, chunk)| {
        let mut col = (fragments[i] / total_aa_samples as f32).max(Vec3A::ZERO);
        col *= Vec3A::splat(2.0).powf(exposure);
        col = somewhat_boring_display_transform(col);
        col = col.powf(1.7); // contrast
        let luma = Vec3A::splat(col.dot(vec3a(0.2126, 0.7152, 0.0722)));
        col = luma * -0.1 + col * 1.1; // saturation
        let c = (col.clamp(Vec3A::ZERO, Vec3A::ONE) * 255.0).as_uvec3();
        chunk.copy_from_slice(&[c.x as u8, c.y as u8, c.z as u8, 255]);
    });

    img.save(format!("demoscene_{}_rend.png", seed))
        .expect("Failed to save image");
}

mod sky {
    use std::f32::consts::PI;

    use glam::{vec3a, Vec3A};

    use obvhs::test_util::sampling::smoothstep;

    use crate::SUN_ANGULAR_DIAMETER;

    // Based on https://github.com/Tw1ddle/Sky-Shader/
    pub struct Sky {
        pub depolarization_factor: f32,
        pub mie_coefficient: f32,
        pub mie_directional_g: f32,
        pub mie_k_coefficient: Vec3A,
        pub mie_v: f32,
        pub mie_zenith_length: f32,
        pub num_molecules: f32,
        pub primaries: Vec3A,
        pub rayleigh: f32,
        pub rayleigh_zenith_length: f32,
        pub refractive_index: f32,
        pub sun_angular_diameter: f32,
        pub sun_intensity_factor: f32,
        pub sun_intensity_falloff_steepness: f32,
        pub turbidity: f32,
        pub sun_position: Vec3A,
    }

    impl Sky {
        pub fn red_sunset(sun_position: Vec3A) -> Sky {
            Sky {
                depolarization_factor: 0.02,
                mie_coefficient: 0.005,
                mie_directional_g: 0.82,
                mie_k_coefficient: vec3a(0.686, 0.678, 0.666),
                mie_v: 3.936,
                mie_zenith_length: 34000.0,
                num_molecules: 2.542e25,
                primaries: vec3a(6.8e-7f32, 5.5e-7f32, 4.5e-7f32),
                rayleigh: 2.28,
                rayleigh_zenith_length: 8400.0,
                refractive_index: 1.00029,
                sun_angular_diameter: SUN_ANGULAR_DIAMETER,
                sun_intensity_factor: 1000.0,
                sun_intensity_falloff_steepness: 1.1,
                turbidity: 4.7,
                sun_position,
            }
        }

        pub fn render(&self, dir: Vec3A) -> Vec3A {
            let sunfade = 1.0 - (1.0 - (self.sun_position.y / 450000.0).exp()).clamp(0.0, 1.0);
            let rayleigh_coefficient = self.rayleigh - (1.0 * (1.0 - sunfade));
            let beta_r = self.total_rayleigh(self.primaries) * rayleigh_coefficient;

            let beta_m = self.total_mie(self.primaries) * self.mie_coefficient;

            let zenith_angle = (0.0f32.max(Vec3A::Y.dot(dir))).acos();
            let denom =
                zenith_angle.cos() + 0.15 * (93.885 - ((zenith_angle * 180.0) / PI)).powf(-1.253);
            let s_r = self.rayleigh_zenith_length / denom;
            let s_m = self.mie_zenith_length / denom;

            let fex = (-(beta_r * s_r + beta_m * s_m)).exp();

            let sun_direction = self.sun_position.normalize();
            let cos_theta = dir.dot(sun_direction);
            let beta_r_theta = beta_r * Self::rayleigh_phase(cos_theta * 0.5 + 0.5);
            let beta_m_theta =
                beta_m * Self::henyey_greenstein_phase(cos_theta, self.mie_directional_g);

            let sun_e = self.sun_intensity(sun_direction.dot(Vec3A::Y));
            let mut lin =
                (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * (1.0 - fex))
                    .powf(1.5);
            lin *= Vec3A::splat(1.0).lerp(
                (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * fex).powf(0.5),
                (1.0 - Vec3A::Y.dot(sun_direction))
                    .powf(5.0)
                    .clamp(0.0, 1.0),
            );

            let sun_angular_diameter_cos = (self.sun_angular_diameter).cos();
            let sundisk = smoothstep(
                sun_angular_diameter_cos,
                sun_angular_diameter_cos, // + 0.00002
                cos_theta,
            );
            let mut l0 = Vec3A::splat(0.1) * fex;
            l0 += sun_e * 19000.0 * fex * sundisk;
            let mut color = (lin + l0) * 0.04;
            let low_falloff = (Vec3A::Y.dot(dir) + 0.4).powf(5.0).max(0.0);
            color = (color * 0.1).powf(3.0) * low_falloff;
            color.powf(1.0 / (1.2 + (1.2 * sunfade))) * 0.5
        }

        fn total_rayleigh(&self, lambda: Vec3A) -> Vec3A {
            (8.0 * PI.powi(3)
                * (self.refractive_index.powi(2) - 1.0).powi(2)
                * (6.0 + 3.0 * self.depolarization_factor))
                / (3.0
                    * self.num_molecules
                    * lambda.powf(4.0)
                    * (6.0 - 7.0 * self.depolarization_factor))
        }

        fn total_mie(&self, lambda: Vec3A) -> Vec3A {
            let c = 0.2 * self.turbidity * 10e-18;
            0.434 * c * PI * (2.0 * PI / lambda).powf(self.mie_v - 2.0) * self.mie_k_coefficient
        }

        fn rayleigh_phase(cos_theta: f32) -> f32 {
            (3.0 / (16.0 * PI)) * (1.0 + cos_theta.powi(2))
        }

        fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
            (1.0 / (4.0 * PI))
                * ((1.0 - g.powi(2)) / (1.0 - 2.0 * g * cos_theta + g.powi(2)).powf(1.5))
        }

        fn sun_intensity(&self, zenith_angle_cos: f32) -> f32 {
            let cutoff_angle = PI / 1.95;
            self.sun_intensity_factor
                * 0.0f32.max(
                    1.0 - (-((cutoff_angle - zenith_angle_cos.acos()).exp()
                        / self.sun_intensity_falloff_steepness)),
                )
        }
    }
}
