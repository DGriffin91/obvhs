use std::time::{Duration, Instant};

use crate::{
    aabb::Aabb,
    bvh2::reinsertion::ReinsertionOptimizer,
    cwbvh::{bvh2_to_cwbvh::bvh2_to_cwbvh, CwBvh},
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
    let mut aabbs = Vec::with_capacity(triangles.len());
    let mut indices = Vec::with_capacity(triangles.len());
    let mut largest_half_area = 0.0;
    let mut avg_half_area = 0.0;

    for (i, tri) in triangles.iter().enumerate() {
        let a = tri.v0;
        let b = tri.v1;
        let c = tri.v2;
        let mut aabb = Aabb::empty();
        aabb.extend(a).extend(b).extend(c);
        let half_area = aabb.half_area();
        largest_half_area = half_area.max(largest_half_area);
        avg_half_area += half_area;
        aabbs.push(aabb);
        indices.push(i as u32);
    }
    avg_half_area /= triangles.len() as f32;

    let start_time = Instant::now();

    if config.pre_split {
        split_aabbs_preset(
            &mut aabbs,
            &mut indices,
            triangles,
            avg_half_area,
            largest_half_area,
        );
    }

    let mut bvh2 = config.ploc_search_distance.build(
        &aabbs,
        indices,
        config.sort_precision,
        config.search_depth_threshold,
    );
    bvh2.uses_spatial_splits = config.pre_split;
    ReinsertionOptimizer::run(&mut bvh2, config.reinsertion_batch_ratio, None);
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
    let cwbvh = bvh2_to_cwbvh(&bvh2, config.max_prims_per_leaf.clamp(1, 3), true, false);

    #[cfg(debug_assertions)]
    {
        bvh2.validate(&aabbs, false, config.pre_split);
        cwbvh.validate(&aabbs, false);
    }

    *core_build_time += start_time.elapsed();

    cwbvh
}
