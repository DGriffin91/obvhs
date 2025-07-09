//! Split large triangles into multiple smaller Aabbs.

use glam::Vec3A;

use crate::{aabb::Aabb, triangle::Triangle};

/// Splits large triangles into multiple smaller Aabbs. Fits the new aabbs tightly to the triangle.
/// Note: This will result in more aabbs than triangles. The indices Vec will have grow with the
/// added Aabb's with the respective mapping back to the initial list of triangles.
/// # Arguments
/// * `avg_half_area` - The average half area of the Triangles
/// * `largest_half_area` - The largest half area of the Triangles
///   This is tuned to try to create splits conservatively enough that it generally
///   wont result in lower traversal performance across a variety of scenes.
///   (Naive splitting can result in lower traversal performance in some scenes)
pub fn split_aabbs_preset(
    aabbs: &mut Vec<Aabb>,
    indices: &mut Vec<u32>,
    triangles: &[Triangle],
    avg_half_area: f32,
    largest_half_area: f32,
) {
    split_aabbs_precise(
        aabbs,
        indices,
        triangles,
        avg_half_area * 3.0,
        (avg_half_area * 4.0).max(avg_half_area * 0.9 + largest_half_area * 0.1),
        1.8,
        1.6,
        12,
        12,
    );
}

/// Splits large triangles into multiple smaller Aabbs. Fits the new aabbs tightly to the triangle.
/// Note: This will result in more aabbs than triangles. The indices Vec will have grow with the
/// added Aabb's with the respective mapping back to the initial list of triangles.
/// # Arguments
/// * `area_thresh_low` - Triangles with aabb half areas below this will not be considered for splitting.
/// * `area_thresh_high` - If the low split factor condition is not met then area_thresh_high > old_cost
///   must be met in addition to best_cost * split_factor_high < old_cost in order for the split to occur
/// * `split_factor_low` - If the resulting smallest aabb half area (best_cost) multiplied by this factor is
///   lower than the original cost the best split will be used (best_cost * split_factor_low < old_cost)
///   (area_thresh_high > old_cost && best_cost * split_factor_high < old_cost)
/// * `max_iterations` - Number of times to evaluate the entire set of aabbs/triangles (including the newly added splits)
/// * `split_tests` - Number of places try splitting the triangle at.
#[allow(clippy::too_many_arguments)]
pub fn split_aabbs_precise(
    aabbs: &mut Vec<Aabb>,
    indices: &mut Vec<u32>,
    triangles: &[Triangle],
    area_thresh_low: f32,
    area_thresh_high: f32,
    split_factor_low: f32,
    split_factor_high: f32,
    max_iterations: u32,
    split_tests: u32,
) {
    crate::scope!("split_aabbs_precise");

    let mut candidates = Vec::new();

    for (i, aabb) in aabbs.iter().enumerate() {
        if aabb.half_area() > area_thresh_low {
            candidates.push(i)
        }
    }

    let mut old_candidates_len = candidates.len();
    for _ in 0..max_iterations {
        for i in 0..candidates.len() {
            let aabb = &mut aabbs[candidates[i]];
            let index = indices[candidates[i]];
            let axis: usize = aabb.largest_axis();

            let tri = triangles[index as usize];

            let mut best_cost = f32::MAX;
            let mut left = *aabb;
            let mut right = *aabb;

            // TODO optimization: create multiple splits simultaneously
            for i in 1..split_tests {
                let n = i as f32 / split_tests as f32;
                let pos = aabb.min[axis] * n + aabb.max[axis] * (1.0 - n);

                let mut tmp_left = *aabb;
                let mut tmp_right = *aabb;

                tmp_left.max[axis] = pos;
                tmp_right.min[axis] = pos;
                let verts = [tri.v0, tri.v1, tri.v2, tri.v0];
                let (t_left, t_right) = split_triangle(axis as u32, pos, verts);
                tmp_left = t_left.intersection(&tmp_left);
                tmp_right = t_right.intersection(&tmp_right);
                let area = tmp_left.half_area() + tmp_right.half_area();
                if area < best_cost {
                    best_cost = area;
                    left = tmp_left;
                    right = tmp_right;
                }
            }

            let old_cost = aabb.half_area();

            if (area_thresh_high > old_cost && best_cost * split_factor_high < old_cost)
                || best_cost * split_factor_low < old_cost
            {
                *aabb = left;
                candidates.push(aabbs.len());
                aabbs.push(right);
                indices.push(index);
            }
        }
        if old_candidates_len == candidates.len() {
            break;
        } else {
            candidates.retain(|c| aabbs[*c].half_area() > area_thresh_low);
            old_candidates_len = candidates.len();
        }
    }
}

/// Based on <https://github.com/embree/embree/blob/be0accfd0b246e2b03355b8ee7710a22c1b49240/kernels/builders/splitter.h#L17C1-L49C6>,
/// but with the "current bounds" moved out.
pub fn split_triangle(dim: u32, pos: f32, v: [Vec3A; 4]) -> (Aabb, Aabb) {
    let mut left = Aabb::INVALID;
    let mut right = Aabb::INVALID;

    // Clip triangle to left and right box by processing all edges
    for i in 0..3 {
        let v0 = v[i];
        let v1 = v[i + 1];
        let v0d = v0[dim as usize];
        let v1d = v1[dim as usize];

        if v0d <= pos {
            // This point is on left side
            left.extend(v0);
        }
        if v0d >= pos {
            // This point is on right side
            right.extend(v0);
        }

        // The edge crosses the splitting location
        if (v0d < pos && pos < v1d) || (v1d < pos && pos < v0d) {
            debug_assert!((v1d - v0d) != 0.0);
            let inv_length = 1.0 / (v1d - v0d);
            let c = Vec3A::mul_add(Vec3A::splat((pos - v0d) * inv_length), v1 - v0, v0);
            left.extend(c);
            right.extend(c);
        }
    }

    (left, right)
}
