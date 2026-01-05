use std::time::Duration;

use glam::*;
use obvhs::{
    BvhBuildParams,
    bvh2::builder::build_bvh2_from_tris,
    ray::{Ray, RayHit},
    test_util::geometry::{PLANE, icosphere},
    triangle::Triangle,
};

fn main() {
    // Build a scene with an icosphere and a plane
    // BVH primitives do not need to be triangles, the BVH builder is only concerned with AABBs.
    // (With the exception of optional precise triangle aabb splitting)
    let mut tris: Vec<Triangle> = Vec::new();
    tris.extend(icosphere(1));
    tris.extend(PLANE);

    // Build the BVH.
    // build_bvh_from_tris is just a helper that can build from BvhBuildParams and the
    // respective presets. Feel free to copy the contents of build_bvh_from_tris or build_bvh.
    // They are very straightforward. If you don't want to use Triangles as the primitive, use
    // build_bvh instead. build_cwbvh_from_tris just adds support for splitting tris.
    let bvh = build_bvh2_from_tris(
        &tris,
        BvhBuildParams::medium_build(),
        &mut Duration::default(),
    );

    // Create a new ray
    let ray = Ray::new_inf(vec3a(0.1, 0.1, 4.0), vec3a(0.0, 0.0, -1.0));

    // Traverse the BVH, finding the closest hit.
    let mut ray_hit = RayHit::none();
    if bvh.ray_traverse(ray, &mut ray_hit, |ray, id| {
        // Use primitive_indices to look up the original primitive id.
        // (Could reorder tris per bvh.primitive_indices to avoid this lookup, see cornell_box_cwbvh example)
        tris[bvh.primitive_indices[id] as usize].intersect(ray)
    }) {
        println!(
            "Hit Triangle {}",
            bvh.primitive_indices[ray_hit.primitive_id as usize]
        );
        println!("Distance to hit: {}", ray_hit.t);
    } else {
        println!("Miss");
    }
}
