use std::time::{Duration, Instant};

use crate::{
    bvh2::reinsertion::ReinsertionOptimizer,
    cwbvh::{bvh2_to_cwbvh::bvh2_to_cwbvh, CwBvh},
    ploc::PlocBuilder,
    splits::split_aabbs_preset,
    triangle::Triangle,
    Boundable, BvhBuildParams,
};

/// Build a cwbvh from the given list of Triangles.
/// Just a helper function / example, feel free to reimplement for your specific use case.
///
/// # Arguments
/// * `triangles` - A list of Triangles.
/// * `config` - Parameters for configuring the BVH building.
/// * `core_build_time` - The core BVH build time. Does not include things like initial AABB
///   generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
pub fn build_cwbvh_from_tris(
    triangles: &[Triangle],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> CwBvh {
    let mut bvh2;
    let start_time;
    if config.pre_split {
        let mut aabbs = Vec::with_capacity(triangles.len());
        let mut indices = Vec::with_capacity(triangles.len());
        let mut largest_half_area = 0.0;
        let mut avg_area = 0.0;

        for (i, tri) in triangles.iter().enumerate() {
            let aabb = tri.aabb();
            let half_area = aabb.half_area();
            largest_half_area = half_area.max(largest_half_area);
            avg_area += half_area;
            aabbs.push(aabb);
            indices.push(i as u32);
        }
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
    ReinsertionOptimizer::default().run(&mut bvh2, config.reinsertion_batch_ratio, None);
    let cwbvh = bvh2_to_cwbvh(&bvh2, config.max_prims_per_leaf.clamp(1, 3), true, false);

    *core_build_time += start_time.elapsed();

    #[cfg(debug_assertions)]
    {
        bvh2.validate(triangles, false, config.pre_split);
        cwbvh.validate(triangles, false);
    }

    cwbvh
}

/// Build a cwbvh from the given list of Boundable primitives.
/// `pre_split` in BvhBuildParams is ignored in this case.
/// Just a helper function / example, feel free to reimplement for your specific use case.
///
/// # Arguments
/// * `primitives` - A list of Primitives that implement Boundable.
/// * `config` - Parameters for configuring the BVH building.
/// * `core_build_time` - The core BVH build time. Does not include things like initial AABB
///   generation or debug validation. This is mostly just here to simplify profiling in [tray_racing](https://github.com/DGriffin91/tray_racing)
// TODO: we could optionally do imprecise basic Aabb splits.
pub fn build_cwbvh<T: Boundable>(
    primitives: &[T],
    config: BvhBuildParams,
    core_build_time: &mut Duration,
) -> CwBvh {
    let start_time = Instant::now();

    let mut bvh2 = PlocBuilder::with_capacity(primitives.len()).build(
        config.ploc_search_distance,
        primitives,
        (0..primitives.len() as u32).collect::<Vec<_>>(),
        config.sort_precision,
        config.search_depth_threshold,
    );
    ReinsertionOptimizer::default().run(&mut bvh2, config.reinsertion_batch_ratio, None);
    let cwbvh = bvh2_to_cwbvh(&bvh2, config.max_prims_per_leaf.clamp(1, 3), true, false);

    #[cfg(debug_assertions)]
    {
        bvh2.validate(primitives, false, config.pre_split);
        cwbvh.validate(primitives, false);
    }

    *core_build_time += start_time.elapsed();

    cwbvh
}
