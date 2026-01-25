/// Traverse a CwBvh with custom node and primitive intersections.
/// I really didn't want to use a macro but it seems like everything else using closures/yielding is slower given
/// both generic node and primitive traversal.
///
/// # Parameters
/// - `$cwbvh`: `&CwBvh` The CwBvh to be traversed.
/// - `$node`: `&CwBvhNode` The current node in the BVH that is being traversed.
/// - `$state`: `Traversal` Mutable traversal state.
/// - `$node_intersection`: An expression that is executed for each node intersection during traversal.
///   It should test for intersection against the current `node`, making use of `state.oct_inv4` u32.
///   It should return a u32 `hitmask` of the node children hitmask corresponding to which nodes were intersected.
/// - `$primitive_intersection`: A code block that is executed for each primitive intersection.
///   It should read the current `state.primitive_id` u32. This is the index into the primitive indices for the
///   current primitive to be tested. Optionally use `break` to halt traversal.
///
/// # Example: Closest hit ray traversal
/// ```
/// use obvhs::{
///     cwbvh::{builder::build_cwbvh_from_tris, node::CwBvhNode},
///     ray::{Ray, RayHit},
///     test_util::geometry::{icosphere, PLANE},
///     triangle::Triangle,
///     BvhBuildParams,
///     traverse,
/// };
/// use glam::*;
/// use std::time::Duration;
///
/// let mut tris: Vec<Triangle> = Vec::new();
/// tris.extend(icosphere(1));
/// tris.extend(PLANE);
///
/// let ray = Ray::new_inf(vec3a(0.1, 0.1, 4.0), vec3a(0.0, 0.0, -1.0));
///
/// let bvh = build_cwbvh_from_tris(&tris, BvhBuildParams::medium_build(), &mut Duration::default());
/// let mut hit = RayHit::none();
/// let mut traverse_ray = ray.clone();
/// let mut state = bvh.new_traversal(ray.direction);
/// let mut node;
/// traverse!(bvh, node, state,
///     // Node intersection:
///     node.intersect_ray(&traverse_ray, state.oct_inv4),
///     // Primitive intersection:
///     {
///         let t = tris[bvh.primitive_indices[state.primitive_id as usize] as usize].intersect(&traverse_ray);
///         if t < traverse_ray.tmax {
///             hit.primitive_id = state.primitive_id;
///             hit.t = t;
///             traverse_ray.tmax = t;
///         }
///     }
/// );
///
/// let did_hit = hit.t < ray.tmax;
/// assert!(did_hit);
/// assert!(bvh.primitive_indices[hit.primitive_id as usize] == 62);
/// ```
#[macro_export]
macro_rules! traverse {
    ($cwbvh:expr, $node:expr, $state:expr, $node_intersection:expr, $primitive_intersection:expr) => {{
        use $crate::faststack::FastStack;
        loop {
            // While the primitive group is not empty
            while $state.primitive_group.y != 0 {
                let local_primitive_index = $crate::cwbvh::firstbithigh($state.primitive_group.y);

                // Remove primitive from current_group
                $state.primitive_group.y &= !(1u32 << local_primitive_index);

                $state.primitive_id = $state.primitive_group.x + local_primitive_index;
                $primitive_intersection
            }
            $state.primitive_group = UVec2::ZERO;

            // If there's remaining nodes in the current group to check
            if $state.current_group.y & 0xff000000 != 0 {
                let hits_imask = $state.current_group.y;

                let child_index_offset = $crate::cwbvh::firstbithigh(hits_imask);
                let child_index_base = $state.current_group.x;

                // Remove node from current_group
                $state.current_group.y &= !(1u32 << child_index_offset);

                // If the node group is not yet empty, push it on the stack
                if $state.current_group.y & 0xff000000 != 0 {
                    $state.stack.push($state.current_group);
                }

                let slot_index = (child_index_offset - 24) ^ ($state.oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();

                let child_node_index = child_index_base + relative_index;

                $node = &$cwbvh.nodes[child_node_index as usize];

                $state.hitmask = $node_intersection;

                $state.current_group.x = $node.child_base_idx;
                $state.primitive_group.x = $node.primitive_base_idx;

                $state.current_group.y = (&$state.hitmask & 0xff000000u32) | ($node.imask as u32);
                $state.primitive_group.y = &$state.hitmask & 0x00ffffffu32;
            } else {
                // Below is only needed when using triangle postponing, which would only be helpful on the
                // GPU (it helps reduce thread divergence). Also, this isn't compatible with traversal yielding.
                // $state.primitive_group = $state.current_group;
                $state.current_group = UVec2::ZERO;
            }

            // If there's no remaining nodes in the current group to check, pop it off the stack.
            if $state.primitive_group.y == 0 && ($state.current_group.y & 0xff000000) == 0 {
                // If the stack is empty, end traversal.
                if $state.stack.is_empty() {
                    #[allow(unused)]
                    {
                        $state.current_group.y = 0;
                    }
                    break;
                }

                $state.current_group = $state.stack.pop_fast();
            }
        }
    }};
}
