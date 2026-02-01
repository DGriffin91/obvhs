#[cfg(test)]
mod tests {

    use std::time::Duration;

    use glam::*;
    use obvhs::{
        BvhBuildParams,
        aabb::Aabb,
        bvh2::builder::{build_bvh2, build_bvh2_from_tris},
        cwbvh::{
            builder::{build_cwbvh, build_cwbvh_from_tris},
            bvh2_to_cwbvh::bvh2_to_cwbvh,
        },
        ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
        ray::{Ray, RayHit},
        test_util::{
            geometry::{demoscene, height_to_triangles, icosphere},
            sampling::{hash_noise, uniform_sample_sphere},
        },
        traverse,
        triangle::Triangle,
    };

    const BUILD_PARAM_SET: [BvhBuildParams; 6] = [
        BvhBuildParams::fastest_build(),
        BvhBuildParams::very_fast_build(),
        BvhBuildParams::fast_build(),
        BvhBuildParams::medium_build(),
        BvhBuildParams::slow_build(),
        BvhBuildParams::very_slow_build(),
    ];

    #[test]
    pub fn build_bvh2_with_empty_aabb() {
        for build_param in BUILD_PARAM_SET {
            let bvh = build_bvh2(&[Aabb::empty()], build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_bvh2_with_many_inf() {
        for build_param in BUILD_PARAM_SET {
            let bvh = build_bvh2(&[Aabb::INFINITY; 10], build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_bvh2_with_many_max() {
        for build_param in BUILD_PARAM_SET {
            let bvh = build_bvh2(&[Aabb::LARGEST; 10], build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_cwbvh_with_empty_aabb() {
        for build_param in BUILD_PARAM_SET {
            let bvh = build_cwbvh(&[Aabb::empty()], build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_bvh2_with_nothing() {
        for build_param in BUILD_PARAM_SET {
            let aabbs: Vec<Aabb> = Vec::new();
            let bvh = build_bvh2(&aabbs, build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_cwbvh_with_nothing() {
        for build_param in BUILD_PARAM_SET {
            let aabbs: Vec<Aabb> = Vec::new();
            let bvh = build_cwbvh(&aabbs, build_param, &mut Duration::default());
            let ray = Ray::new_inf(Vec3A::Z, -Vec3A::Z);
            assert!(!bvh.ray_traverse(ray, &mut RayHit::none(), |_ray, _id| f32::INFINITY));
        }
    }

    #[test]
    pub fn build_bvh_with_varying_prim_counts() {
        let mut tris = height_to_triangles(|_x: usize, _y: usize| -> f32 { 0.0 }, 4, 4);
        while tris.len() > 0 {
            tris.pop();
            for build_param in BUILD_PARAM_SET {
                let bvh = build_bvh2_from_tris(&tris, build_param, &mut Duration::default());
                bvh.validate(&tris, false, !build_param.pre_split);
                let bvh = build_cwbvh_from_tris(&tris, build_param, &mut Duration::default());
                bvh.validate(&tris, false);
            }
        }
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

    #[allow(clippy::too_many_arguments)]
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
        let cwbvh = build_cwbvh_from_tris(
            tris,
            BvhBuildParams::medium_build(),
            &mut Duration::default(),
        );

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
                if cwbvh.ray_traverse(ray, &mut hit, |ray, id| bvh_tris[id].intersect(ray)) {
                    eval(x, y, hit);
                }
            }
        }
    }

    #[test]
    pub fn traverse_aabb() {
        let tris = demoscene(201, 0);
        let aabb = Aabb::new(vec3a(0.511, -1.0, 0.511), vec3a(0.611, 1.0, 0.611));

        let mut refrence_intersect_sum = 0usize;
        let mut refrence_count = 0;
        for (primitive_id, tri) in tris.iter().enumerate() {
            if aabb.intersect_aabb(&tri.aabb()) {
                refrence_intersect_sum = refrence_intersect_sum.wrapping_add(primitive_id);
                refrence_count += 1;
            }
        }

        // Bvh2
        let bvh2 = build_bvh2_from_tris(
            &tris,
            BvhBuildParams::fast_build(),
            &mut Duration::default(),
        );
        let mut intersect_sum = 0usize;
        let mut intersect_count = 0;
        bvh2.validate(&tris, false, false);
        bvh2.aabb_traverse(aabb, |bvh, id| {
            let node = &bvh.nodes[id as usize];
            for i in 0..node.prim_count {
                let primitive_id = bvh.primitive_indices[(node.first_index + i) as usize] as usize;
                let tri = tris[primitive_id];
                if aabb.intersect_aabb(&tri.aabb()) {
                    intersect_count += 1;
                    intersect_sum = intersect_sum.wrapping_add(primitive_id);
                }
            }
            true
        });
        assert_eq!(refrence_count, intersect_count);
        assert_eq!(refrence_intersect_sum, intersect_sum);

        // CwBvh
        let cwbvh = build_cwbvh_from_tris(
            &tris,
            BvhBuildParams::fast_build(),
            &mut Duration::default(),
        );
        let mut cw_intersect_count = 0;
        let mut cw_intersect_sum = 0usize;
        cwbvh.validate(&tris, false);

        let mut state = cwbvh.new_traversal(Vec3A::ZERO);
        let mut node;
        traverse!(
            cwbvh,
            node,
            state,
            node.intersect_aabb(&aabb, state.oct_inv4),
            {
                let primitive_id = cwbvh.primitive_indices[state.primitive_id as usize] as usize;
                let tri = tris[primitive_id];
                if aabb.intersect_aabb(&tri.aabb()) {
                    cw_intersect_count += 1;
                    cw_intersect_sum = cw_intersect_sum.wrapping_add(primitive_id);
                }
            }
        );

        assert_eq!(refrence_count, cw_intersect_count);
        assert_eq!(refrence_intersect_sum, cw_intersect_sum);
    }

    #[test]
    pub fn traverse_point() {
        let tris = icosphere(0);

        // CwBvh
        let cwbvh = build_cwbvh_from_tris(
            &tris,
            BvhBuildParams::fast_build(),
            &mut Duration::default(),
        );
        cwbvh.validate(&tris, false);

        // Bvh2
        let bvh2 = build_bvh2_from_tris(
            &tris,
            BvhBuildParams::fast_build(),
            &mut Duration::default(),
        );
        bvh2.validate(&tris, false, true);

        for i in 0..512 {
            let point =
                uniform_sample_sphere(vec2(hash_noise(uvec2(0, 0), i), hash_noise(uvec2(0, 1), i)));

            let mut refrence_intersect_sum = 0usize;
            let mut refrence_count = 0;
            for (primitive_id, tri) in tris.iter().enumerate() {
                if tri.aabb().contains_point(point) {
                    refrence_intersect_sum = refrence_intersect_sum.wrapping_add(primitive_id);
                    refrence_count += 1;
                }
            }

            let mut cw_intersect_count = 0;
            let mut cw_intersect_sum = 0usize;
            let mut state = cwbvh.new_traversal(Vec3A::ZERO);
            let mut node;
            traverse!(
                cwbvh,
                node,
                state,
                node.contains_point(&point, state.oct_inv4),
                {
                    let primitive_id =
                        cwbvh.primitive_indices[state.primitive_id as usize] as usize;
                    let tri = tris[primitive_id];
                    if tri.aabb().contains_point(point) {
                        cw_intersect_count += 1;
                        cw_intersect_sum = cw_intersect_sum.wrapping_add(primitive_id);
                    }
                }
            );

            assert_eq!(refrence_count, cw_intersect_count);
            assert_eq!(refrence_intersect_sum, cw_intersect_sum);

            let mut intersect_count = 0;
            let mut intersect_sum = 0usize;
            bvh2.point_traverse(point, |bvh, id| {
                let node = &bvh.nodes[id as usize];
                for i in 0..node.prim_count {
                    let primitive_id =
                        bvh.primitive_indices[(node.first_index + i) as usize] as usize;
                    let tri = tris[primitive_id];
                    if tri.aabb().contains_point(point) {
                        intersect_count += 1;
                        intersect_sum = intersect_sum.wrapping_add(primitive_id);
                    }
                }
                true
            });

            assert_eq!(refrence_count, intersect_count);
            assert_eq!(refrence_intersect_sum, intersect_sum);
        }
    }

    #[test]
    pub fn compute_parents_cwbvh() {
        let tris = demoscene(100, 0);
        let cwbvh = build_cwbvh_from_tris(
            &tris,
            BvhBuildParams::fast_build(),
            &mut Duration::default(),
        );
        cwbvh.validate(&tris, false);
        let parents = cwbvh.compute_parents();
        for (child, parent) in parents.iter().enumerate().skip(1) {
            let node = cwbvh.nodes[*parent as usize];
            let mut found_child = false;
            for ch in 0..8 {
                if !node.is_leaf(ch) {
                    let child_index = node.child_node_index(ch);
                    if child_index as usize == child {
                        found_child = true;
                        break;
                    }
                }
            }
            assert!(found_child, "child{child}, parent{parent}");
        }
    }

    #[test]
    pub fn order_children_cwbvh() {
        let tris = demoscene(100, 0);
        let triangles: &[Triangle] = &tris;
        let mut aabbs = Vec::with_capacity(triangles.len());

        let config = BvhBuildParams::very_fast_build();
        let mut indices = Vec::with_capacity(triangles.len());
        for (i, tri) in triangles.iter().enumerate() {
            let a = tri.v0;
            let b = tri.v1;
            let c = tri.v2;
            let mut aabb = Aabb::empty();
            aabb.extend(a).extend(b).extend(c);
            aabbs.push(aabb);
            indices.push(i as u32);
        }

        let bvh2 = PlocBuilder::new().build(
            config.ploc_search_distance,
            &aabbs,
            indices,
            config.sort_precision,
            config.search_depth_threshold,
        );
        let mut cwbvh = bvh2_to_cwbvh(&bvh2, config.max_prims_per_leaf.clamp(1, 3), true, false);

        cwbvh.validate(&tris, false);
        for node in 0..cwbvh.nodes.len() {
            cwbvh.order_node_children(&aabbs, node, false);
        }
        cwbvh.validate(&tris, false);
        cwbvh.order_children(&aabbs, false);
        cwbvh.validate(&tris, false);
    }

    #[test]
    pub fn exact_aabbs_cwbvh() {
        let tris = demoscene(100, 0);
        let triangles: &[Triangle] = &tris;
        let mut aabbs = Vec::with_capacity(triangles.len());

        let config = BvhBuildParams::very_fast_build();
        let mut indices = Vec::with_capacity(triangles.len());
        for (i, tri) in triangles.iter().enumerate() {
            let a = tri.v0;
            let b = tri.v1;
            let c = tri.v2;
            let mut aabb = Aabb::empty();
            aabb.extend(a).extend(b).extend(c);
            aabbs.push(aabb);
            indices.push(i as u32);
        }

        let bvh2 = PlocBuilder::new().build(
            config.ploc_search_distance,
            &aabbs,
            indices,
            config.sort_precision,
            config.search_depth_threshold,
        );
        let mut cwbvh = bvh2_to_cwbvh(&bvh2, config.max_prims_per_leaf.clamp(1, 3), true, true);

        if let Some(exact_node_aabbs) = &cwbvh.exact_node_aabbs {
            for node in &cwbvh.nodes {
                for ch in 0..8 {
                    if !node.is_leaf(ch) {
                        let child_node_index = node.child_node_index(ch) as usize;
                        let compressed_aabb = node.child_aabb(ch);
                        let child_node_self_compressed_aabb = cwbvh.nodes[child_node_index].aabb();
                        let exact_aabb = &exact_node_aabbs[child_node_index];

                        assert!(exact_aabb.min.cmpge(compressed_aabb.min).all());
                        assert!(exact_aabb.max.cmple(compressed_aabb.max).all());
                        assert!(
                            exact_aabb
                                .min
                                .cmpge(child_node_self_compressed_aabb.min)
                                .all()
                        );
                        assert!(
                            exact_aabb
                                .max
                                .cmple(child_node_self_compressed_aabb.max)
                                .all()
                        );
                    }
                }
            }
        }

        cwbvh.order_children(&aabbs, false);
        cwbvh.validate(&tris, false);
        cwbvh.order_children(&aabbs, false);
        cwbvh.validate(&tris, false);
    }

    #[test]
    pub fn reuse_allocs() {
        fn aabbs_and_indices(tris: &Vec<Triangle>) -> (Vec<Aabb>, Vec<u32>) {
            let mut aabbs = Vec::with_capacity(tris.len());
            let mut indices = Vec::with_capacity(tris.len());
            let mut largest_half_area = 0.0;

            for (i, tri) in tris.iter().enumerate() {
                let a = tri.v0;
                let b = tri.v1;
                let c = tri.v2;
                let mut aabb = Aabb::empty();
                aabb.extend(a).extend(b).extend(c);
                let half_area = aabb.half_area();
                largest_half_area = half_area.max(largest_half_area);
                aabbs.push(aabb);
                indices.push(i as u32);
            }
            (aabbs, indices)
        }

        let tris = demoscene(99, 0);
        let (aabbs, indices) = aabbs_and_indices(&tris);

        let mut builder = PlocBuilder::with_capacity(aabbs.len());

        let mut bvh2 = builder.build(
            PlocSearchDistance::default(),
            &aabbs,
            indices,
            SortPrecision::U64,
            0,
        );

        bvh2.validate(&tris, false, true);

        let tris = demoscene(98, 0);
        let (aabbs, indices) = aabbs_and_indices(&tris);

        builder.build_with_bvh(
            &mut bvh2,
            PlocSearchDistance::default(),
            &aabbs,
            indices,
            SortPrecision::U64,
            0,
        );

        bvh2.validate(&tris, false, true);
    }
}
