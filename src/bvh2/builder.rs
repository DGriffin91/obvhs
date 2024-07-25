use std::time::{Duration, Instant};

use crate::{
    aabb::Aabb, splits::split_aabbs_preset, triangle::Triangle, Boundable, BvhBuildParams,
};

use super::{leaf_collapser::collapse, reinsertion::ReinsertionOptimizer, Bvh2};

/// Build a bvh2 from the given list of Triangles.
/// Just a helper function / example, feel free to reimplement for your specific use case.
///
/// # Arguments
/// * `triangles` - A list of Triangles.
/// * `config` - Parameters for configuring the BVH building.
/// * `core_build_time` - The core BVH build time. Does not include things like initial AABB
/// generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
pub fn build_bvh2_from_tris(
    triangles: &[Triangle],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> Bvh2 {
    let mut aabbs = Vec::with_capacity(triangles.len());
    let mut indices = Vec::with_capacity(triangles.len());
    let mut largest_half_area = 0.0;
    let mut avg_area = 0.0;

    for (i, tri) in triangles.iter().enumerate() {
        let a = tri.v0;
        let b = tri.v1;
        let c = tri.v2;
        let mut aabb = Aabb::empty();
        aabb.extend(a).extend(b).extend(c);
        let half_area = aabb.half_area();
        largest_half_area = half_area.max(largest_half_area);
        avg_area += half_area;
        aabbs.push(aabb);
        indices.push(i as u32);
    }
    avg_area /= triangles.len() as f32;

    let start_time = Instant::now();

    if config.pre_split {
        split_aabbs_preset(
            &mut aabbs,
            &mut indices,
            triangles,
            avg_area,
            largest_half_area,
        );
    }

    let mut bvh2 = config.ploc_search_distance.build(
        &aabbs,
        indices,
        config.sort_precision,
        config.search_depth_threshold,
    );
    ReinsertionOptimizer::run(&mut bvh2, config.reinsertion_batch_ratio, None);
    collapse(
        &mut bvh2,
        config.max_prims_per_leaf,
        config.collapse_traversal_cost,
    );
    ReinsertionOptimizer::run(
        &mut bvh2,
        config.reinsertion_batch_ratio * config.post_collapse_reinsertion_batch_ratio_multiplier,
        None,
    );

    *core_build_time += start_time.elapsed();

    #[cfg(debug_assertions)]
    {
        bvh2.validate(triangles, false, config.pre_split);
    }

    bvh2
}

/// Build a bvh2 from the given list of Boundable primitives.
/// `pre_split` in BvhBuildParams is ignored in this case.
/// Just a helper function / example, feel free to reimplement for your specific use case.
///
/// # Arguments
/// * `primitives` - A list of Primitives that implement Boundable.
/// * `config` - Parameters for configuring the BVH building.
/// * `core_build_time` - The core BVH build time. Does not include things like initial AABB
/// generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
// TODO: we could optionally do imprecise basic Aabb splits.
pub fn build_bvh2<T: Boundable>(
    primitives: &[T],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> Bvh2 {
    let mut aabbs = Vec::with_capacity(primitives.len());
    let mut indices = Vec::with_capacity(primitives.len());

    for (i, primitive) in primitives.iter().enumerate() {
        indices.push(i as u32);
        aabbs.push(primitive.aabb());
    }

    let start_time = Instant::now();

    let mut bvh2 = config.ploc_search_distance.build(
        &aabbs,
        indices,
        config.sort_precision,
        config.search_depth_threshold,
    );
    ReinsertionOptimizer::run(&mut bvh2, config.reinsertion_batch_ratio, None);
    collapse(
        &mut bvh2,
        config.max_prims_per_leaf,
        config.collapse_traversal_cost,
    );
    ReinsertionOptimizer::run(
        &mut bvh2,
        config.reinsertion_batch_ratio * config.post_collapse_reinsertion_batch_ratio_multiplier,
        None,
    );

    *core_build_time += start_time.elapsed();

    #[cfg(debug_assertions)]
    {
        bvh2.validate(primitives, false, config.pre_split);
    }

    bvh2
}
