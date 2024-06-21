use std::time::Instant;

use crate::{
    aabb::Aabb, splits::split_aabbs_preset, triangle::Triangle, Boundable, BvhBuildParams,
};

use super::{leaf_collapser::collapse, reinsertion::ReinsertionOptimizer, Bvh2};

/// Build a bvh2 from the given list of Triangles.
pub fn build_bvh2_from_tris(
    triangles: &[Triangle],
    config: BvhBuildParams,
    core_build_time: &mut f32,
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
    collapse(&mut bvh2, config.max_prims_per_leaf);

    *core_build_time += start_time.elapsed().as_secs_f32();

    bvh2
}

/// Build a bvh2 from the given list of Boundable primitives.
/// `pre_split` in BvhBuildParams is ignored in this case.
// TODO: we could optionally do imprecise basic Aabb splits.
pub fn build_bvh2<T: Boundable>(
    primitives: &[T],
    config: BvhBuildParams,
    core_build_time: &mut f32,
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
    collapse(&mut bvh2, config.max_prims_per_leaf);

    *core_build_time += start_time.elapsed().as_secs_f32();

    bvh2
}
