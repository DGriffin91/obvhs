use std::time::{Duration, Instant};

use crate::{
    Boundable, BvhBuildParams, ploc::PlocBuilder, splits::split_aabbs_preset, triangle::Triangle,
};

use super::{Bvh2, leaf_collapser::collapse, reinsertion::ReinsertionOptimizer};

/// Build a bvh2 from the given list of Triangles.
/// Just a helper function / example, feel free to reimplement for your specific use case.
///
/// # Arguments
/// * `triangles` - A list of Triangles.
/// * `config` - Parameters for configuring the BVH building.
/// * `core_build_time` - The core BVH build time. Does not include things like initial AABB
///   generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
pub fn build_bvh2_from_tris(
    triangles: &[Triangle],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> Bvh2 {
    let mut bvh2;
    let start_time;
    if config.pre_split {
        let mut largest_half_area = 0.0;
        let mut avg_area = 0.0;

        let mut aabbs = triangles
            .iter()
            .map(|tri| {
                let aabb = tri.aabb();
                let half_area = aabb.half_area();
                largest_half_area = half_area.max(largest_half_area);
                avg_area += half_area;
                aabb
            })
            .collect::<Vec<_>>();
        let mut indices = (0..triangles.len() as u32).collect::<Vec<_>>();

        avg_area /= triangles.len() as f32;

        start_time = Instant::now();

        split_aabbs_preset(
            &mut aabbs,
            &mut indices,
            triangles,
            avg_area,
            largest_half_area,
        );
        bvh2 = PlocBuilder::with_capacity(aabbs.len()).build(
            config.ploc_search_distance,
            &aabbs,
            indices,
            config.sort_precision,
            config.search_depth_threshold,
        );
    } else {
        start_time = Instant::now();
        bvh2 = PlocBuilder::with_capacity(triangles.len()).build(
            config.ploc_search_distance,
            triangles,
            (0..triangles.len() as u32).collect::<Vec<_>>(),
            config.sort_precision,
            config.search_depth_threshold,
        );
    }

    bvh2.uses_spatial_splits = config.pre_split;
    let mut reinsertion_optimizer = ReinsertionOptimizer::default();
    reinsertion_optimizer.run(&mut bvh2, config.reinsertion_batch_ratio, None);
    collapse(
        &mut bvh2,
        config.max_prims_per_leaf.clamp(1, 255),
        config.collapse_traversal_cost,
    );
    reinsertion_optimizer.run(
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
///   generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
// TODO: we could optionally do imprecise basic Aabb splits.
pub fn build_bvh2<T: Boundable>(
    primitives: &[T],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> Bvh2 {
    let start_time = Instant::now();

    let mut bvh2 = PlocBuilder::with_capacity(primitives.len()).build(
        config.ploc_search_distance,
        primitives,
        (0..primitives.len() as u32).collect::<Vec<_>>(),
        config.sort_precision,
        config.search_depth_threshold,
    );
    let mut reinsertion_optimizer = ReinsertionOptimizer::default();
    reinsertion_optimizer.run(&mut bvh2, config.reinsertion_batch_ratio, None);
    collapse(
        &mut bvh2,
        config.max_prims_per_leaf.clamp(1, 255),
        config.collapse_traversal_cost,
    );
    reinsertion_optimizer.run(
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
