#[cfg(test)]
mod tests {
    use glam::*;
    use obvhs::{
        aabb::Aabb,
        bvh2::builder::build_bvh2,
        cwbvh::builder::{build_cwbvh, build_cwbvh_from_tris},
        ray::{Ray, RayHit},
        test_util::geometry::height_to_triangles,
        triangle::Triangle,
        BvhBuildParams,
    };

    #[test]
    pub fn build_bvh2_with_empty_aabb() {
        let bvh = build_bvh2(&[Aabb::empty()], BvhBuildParams::medium_build(), &mut 0.0);
        let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
        assert!(!bvh.traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
    }

    #[test]
    pub fn build_cwbvh_with_empty_aabb() {
        let bvh = build_cwbvh(&[Aabb::empty()], BvhBuildParams::medium_build(), &mut 0.0);
        let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
        assert!(!bvh.traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
    }

    #[test]
    pub fn build_bvh2_with_nothing() {
        let aabbs: Vec<Aabb> = Vec::new();
        let bvh = build_bvh2(&aabbs, BvhBuildParams::medium_build(), &mut 0.0);
        let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
        assert!(!bvh.traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
    }

    #[test]
    pub fn build_cwbvh_with_nothing() {
        let aabbs: Vec<Aabb> = Vec::new();
        let bvh = build_cwbvh(&aabbs, BvhBuildParams::medium_build(), &mut 0.0);
        let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
        assert!(!bvh.traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
    }

    #[test]
    pub fn check_flat_subdivided_plane_normals() {
        let tris = height_to_triangles(|_x: usize, _y: usize| -> f32 { 0.0 }, 4, 4);
        let mut hit_count = 0;
        eval_render(
            |_x: u32, _y: u32, hit: RayHit| {
                let n = tris[hit.primitive_id as usize].compute_normal();
                if n == Vec3A::Y {
                    hit_count += 1
                }
            },
            &tris,
            256,
            256,
            90.0f32.to_radians(),
            vec3a(0.0, 0.9, 0.0),
            vec3a(0.0, 0.0, 0.0),
            Vec3A::X,
        );
        assert_eq!(hit_count, 256 * 256)
    }

    pub fn eval_render<F>(
        mut eval: F,
        tris: &[Triangle],
        width: u32,
        height: u32,
        fov: f32,
        eye: Vec3A,
        look_at: Vec3A,
        up: Vec3A,
    ) where
        F: FnMut(u32, u32, RayHit),
    {
        let cwbvh = build_cwbvh_from_tris(tris, BvhBuildParams::medium_build(), &mut 0.0);

        let bvh_tris = cwbvh
            .primitive_indices
            .iter()
            .map(|i| tris[*i as usize])
            .collect::<Vec<Triangle>>();

        let target_size = Vec2::new(width as f32, height as f32);

        // Compute camera projection & view matrices
        let aspect_ratio = target_size.x / target_size.y;
        let proj_inv = Mat4::perspective_infinite_reverse_rh(fov, aspect_ratio, 0.01).inverse();
        let view_inv = Mat4::look_at_rh(eye.into(), look_at.into(), up.into()).inverse();

        for x in 0..width {
            for y in 0..height {
                let frag_coord = uvec2(x, y);
                let mut screen_uv = frag_coord.as_vec2() / target_size;
                screen_uv.y = 1.0 - screen_uv.y;
                let ndc = screen_uv * 2.0 - Vec2::ONE;
                let clip_pos = vec4(ndc.x, ndc.y, 1.0, 1.0);

                let mut vs = proj_inv * clip_pos;
                vs /= vs.w;
                let direction = (Vec3A::from((view_inv * vs).xyz()) - eye).normalize();
                let ray = Ray::new(eye, direction, 0.0, f32::MAX);

                let mut hit = RayHit::none();
                if cwbvh.traverse(ray, &mut hit, |ray, id| bvh_tris[id].intersect(ray)) {
                    eval(x, y, hit);
                }
            }
        }
    }
}
