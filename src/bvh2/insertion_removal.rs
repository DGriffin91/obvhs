use core::f32;

use crate::{
    bvh2::{Bvh2, Bvh2Node},
    heapstack::HeapStack,
    Boundable, INVALID,
};

/// Removes and returns the leaf specified by `node_id`.
/// Puts `node_id` sibling in its parents place then moves the last two nodes in the now empty slots at `node_id` and
/// its sibling.
///
/// Doesn't update the primitive_indices mapping. If this node is just going to be re-inserted again, nothing needs to
/// be done with primitive_indices, the mapping will still be valid. If this primitive needs to be removed permanently,
/// primitive_indices would need to be updated. If nodes have multiple primitives it would probably be best to just
/// rebuild primitive_indices occasionally, cleaning up removed primitives. If not, a free list could be used.
/// primitive_indices would also need to be updated if the primitive index it's pointing to was removed, so this really
/// probably needs to be managed by the user anyway.
///
/// # Arguments
/// * `bvh` - The Bvh2 the new node is being added to
/// * `node_id` - The index into bvh.nodes of the node that is to be removed
/// * `parents` - A mapping from a given node index to that node's parent for each node in the bvh. insert_leaf_node
///     will update this mapping when it inserts a node.
/// * `primitives_to_nodes` - A mapping from primitives back to nodes (may eventually be optionally included in the Bvh2)
pub fn remove_leaf(
    bvh: &mut Bvh2,
    node_id: usize,
    parents: &mut Vec<u32>,
    mut primitives_to_nodes: Option<&mut Vec<u32>>,
) -> Bvh2Node {
    let node_to_remove = bvh.nodes[node_id];

    if bvh.nodes.len() == 1 {
        // Special case if the BVH is just a leaf
        bvh.nodes.clear();
        parents.clear();
        if let Some(primitives_to_nodes) = primitives_to_nodes {
            primitives_to_nodes.clear();
        }
        return node_to_remove;
    }

    if let Some(primitives_to_nodes) = primitives_to_nodes.as_deref_mut() {
        // Invalidate primitives_to_nodes instances
        for prim_id in
            node_to_remove.first_index..node_to_remove.first_index + node_to_remove.prim_count
        {
            primitives_to_nodes[prim_id as usize] = INVALID;
        }
    }

    assert!(node_to_remove.is_leaf());
    let sibling_id = Bvh2Node::get_sibling_id(node_id);
    debug_assert_eq!(parents[node_id], parents[sibling_id]); // Both children should already have the same parent.
    let mut parent_id = parents[node_id] as usize;

    // Put sibling in parent's place (parent doesn't exist anymore)
    bvh.nodes[parent_id] = bvh.nodes[sibling_id];
    let sibling = &mut bvh.nodes[parent_id];
    if sibling.is_leaf() {
        if let Some(primitives_to_nodes) = primitives_to_nodes.as_deref_mut() {
            // Tell primitives where their node went.
            for prim_id in sibling.first_index..sibling.first_index + sibling.prim_count {
                primitives_to_nodes[prim_id as usize] = parent_id as u32;
            }
        }
    } else {
        // Tell children of sibling where their parent went.
        let left_sibling_child = sibling.first_index as usize;
        parents[left_sibling_child] = parent_id as u32;
        parents[left_sibling_child + 1] = parent_id as u32;
    }
    // Don't need to update other parents here since the parent that was for this `parent_id` slot is now the direct
    // parent of the moved sibling, and the parents of `node_id` and `sibling_id` are updated below.

    // Now slots at both node_id and sibling_id are empty.
    // Take the last two nodes "src" and put them in those now empty "dst" slots.
    let end_nodes = node_id >= bvh.nodes.len() - 2;
    if end_nodes {
        // If these were already the last 2 nodes in the list we can just discard both.
        bvh.nodes.pop().unwrap();
        bvh.nodes.pop().unwrap();
        parents.pop().unwrap();
        parents.pop().unwrap();
    } else {
        let dst_left_id = Bvh2Node::get_left_sibling_id(node_id);
        let dst_right_id = Bvh2Node::get_right_sibling_id(node_id);

        let src_left_id = bvh.nodes.len() as u32 - 2;
        let src_right_id = Bvh2Node::get_sibling_id32(src_left_id);
        let src_right_parent = parents.pop().unwrap();
        let src_left_parent = parents.pop().unwrap();

        parents[dst_left_id] = src_left_parent;
        parents[dst_right_id] = src_right_parent;

        debug_assert_eq!(src_left_parent, src_right_parent); // Both children should already have the same parent.
        let parent_of_relocated = &mut bvh.nodes[src_left_parent as usize];
        debug_assert!(!parent_of_relocated.is_leaf());
        debug_assert_eq!(parent_of_relocated.first_index, src_left_id);
        debug_assert_eq!(parent_of_relocated.first_index + 1, src_right_id);
        // Tell the actual parent of the nodes that are moving where they're going to be now.
        bvh.nodes[src_left_parent as usize].first_index = dst_left_id as u32;

        let right_src_sibling = bvh.nodes.pop().unwrap(); // Last node is right src sibling
        if right_src_sibling.is_leaf() {
            if let Some(ref mut primitives_to_nodes) = primitives_to_nodes {
                // Tell primitives where their node went.
                let start = right_src_sibling.first_index;
                let end = right_src_sibling.first_index + right_src_sibling.prim_count;
                for prim_id in start..end {
                    primitives_to_nodes[prim_id as usize] = dst_right_id as u32;
                }
            }
        } else {
            // Go to children of right_src_sibling and tell them where their parent went
            parents[right_src_sibling.first_index as usize] = dst_right_id as u32;
            parents[right_src_sibling.first_index as usize + 1] = dst_right_id as u32;
        }
        bvh.nodes[dst_right_id] = right_src_sibling;

        let left_src_sibling = bvh.nodes.pop().unwrap(); // Last node is left src sibling
        if left_src_sibling.is_leaf() {
            if let Some(ref mut primitives_to_nodes) = primitives_to_nodes {
                // Tell primitives where their node went.
                let start = left_src_sibling.first_index;
                let end = left_src_sibling.first_index + left_src_sibling.prim_count;
                for prim_id in start..end {
                    primitives_to_nodes[prim_id as usize] = dst_left_id as u32;
                }
            }
        } else {
            // Go to children of left_src_sibling and tell them where their parent went
            parents[left_src_sibling.first_index as usize] = dst_left_id as u32;
            parents[left_src_sibling.first_index as usize + 1] = dst_left_id as u32;
        }
        bvh.nodes[dst_left_id] = left_src_sibling;

        // If the to be removed node's parent was at the end of the array and has now moved update parent_id:
        if parent_id as u32 == src_left_id {
            parent_id = dst_left_id;
        }
        if parent_id as u32 == src_right_id {
            parent_id = dst_right_id;
        }
    }

    // Need to work up the tree updating the aabbs since we just removed a node.
    bvh.refit_from_fast(parent_id, &parents);

    // Return the removed node.
    node_to_remove
}

// The index and inherited_cost of a given candidate sibling used for insertion.
#[derive(Debug, Default, Clone, Copy)]
pub struct SiblingInsertionCandidate {
    inherited_cost: f32,
    index: u32,
}

/// Searches through tree recursively to find the best sibling for the node being inserted. The best sibling is
/// classified as the sibling that if chosen it would increase the surface area of the BVH the least.
/// When the best sibling is found, a parent of both the sibling and the new node is put in the location of
/// the sibling and both the sibling and new node are added to the end of the bvh.nodes.
/// See "Branch and Bound" https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf
///
/// # Returns
/// The index of the newly added node (always `bvh.nodes.len() - 1` since the node it put at the end).
///
/// # Arguments
/// * `bvh` - The Bvh2 the new node is being added to
/// * `new_node` - The new node. This node must be a leaf and already have a valid first_index into primitive_indices
/// * `parents` - A mapping from a given node index to that node's parent for each node in the bvh. insert_leaf_node
///     will update this mapping when it inserts a node.
/// * `stack` - Used for the traversal stack. Needs to be large enough to initially accommodate traversal to the
///     deepest leaf of the BVH. insert_leaf_node() will resize this stack after traversal to be at least 2x the required
///     size. This ends up being quite a bit faster than using a Vec and works well when inserting multiple nodes. But does
///     require the user to provide a good initial guess. SiblingInsertionCandidate is tiny so be generous. Something like:
///     `stack.reserve(bvh.depth(0) * 2).max(1000);` If you are inserting a lot of leafs don't call bvh.depth(0) with each
///      leaf just let insert_leaf_node() resize the stack as needed.
/// * `primitives_to_nodes` - A mapping from primitives back to nodes (may eventually be optionally included in the Bvh2)
pub fn insert_leaf_node(
    bvh: &mut Bvh2,
    new_node: Bvh2Node,
    parents: &mut Vec<u32>,
    stack: &mut HeapStack<SiblingInsertionCandidate>,
    mut primitives_to_nodes: Option<&mut Vec<u32>>,
) -> usize {
    assert!(new_node.is_leaf());
    let mut min_cost = f32::MAX;
    let mut best_sibling_candidate_id = 0;
    let mut max_stack_len = 1;

    stack.clear();
    let root_aabb = bvh.nodes[0].aabb;

    // Traverse the BVH to find the best sibling
    stack.push(SiblingInsertionCandidate {
        inherited_cost: root_aabb.union(&new_node.aabb).half_area() - root_aabb.half_area(),
        index: 0,
    });
    while let Some(sibling_candidate) = stack.pop() {
        let current_node_index = sibling_candidate.index as usize;

        let candidate = &bvh.nodes[current_node_index];

        let direct_cost = candidate.aabb.union(&new_node.aabb).half_area();
        let cost_increase = direct_cost - candidate.aabb.half_area();
        let inherited_cost = sibling_candidate.inherited_cost + cost_increase;
        let cost = direct_cost + inherited_cost;

        if cost < min_cost {
            min_cost = cost;
            best_sibling_candidate_id = current_node_index;
            // If this is not a leaf, it's possible a better cost could be found further down.
            if !candidate.is_leaf() {
                stack.push(SiblingInsertionCandidate {
                    inherited_cost,
                    index: candidate.first_index,
                });
                stack.push(SiblingInsertionCandidate {
                    inherited_cost,
                    index: candidate.first_index + 1,
                });
                max_stack_len = stack.len().max(max_stack_len);
            }
        }
    }

    if max_stack_len * 2 > stack.cap() {
        stack.reserve(max_stack_len * 2);
    }

    let best_sibling_candidate = bvh.nodes[best_sibling_candidate_id];

    // To avoid having gaps or re-arranging the BVH:
    // The new parent goes in the sibling's position.
    // The sibling and new node go on the end.
    let new_sibling_id = bvh.nodes.len() as u32;
    let new_parent = Bvh2Node {
        aabb: new_node.aabb.union(&best_sibling_candidate.aabb),
        prim_count: 0,
        first_index: new_sibling_id,
    };

    // New parent goes in the sibling's position.
    let new_parent_id = best_sibling_candidate_id;
    bvh.nodes[new_parent_id] = new_parent;

    if best_sibling_candidate.is_leaf() {
        if let Some(primitives_to_nodes) = primitives_to_nodes.as_deref_mut() {
            // Tell primitives where their node went.
            let start = best_sibling_candidate.first_index;
            let end = best_sibling_candidate.first_index + best_sibling_candidate.prim_count;
            for prim_id in start..end {
                primitives_to_nodes[prim_id as usize] = new_sibling_id;
            }
        }
    } else {
        // If the best selected sibling was an inner node, we need to update the parents mapping so that the children of
        // that node point to the new location that it's being moved to.
        parents[best_sibling_candidate.first_index as usize] = new_sibling_id;
        parents[best_sibling_candidate.first_index as usize + 1] = new_sibling_id;
    }
    bvh.nodes.push(best_sibling_candidate);
    let new_node_id = bvh.nodes.len();
    bvh.nodes.push(new_node); // Put the new node at the very end.
    parents.push(new_parent_id as u32);
    parents.push(new_parent_id as u32);

    if let Some(primitives_to_nodes) = primitives_to_nodes.as_deref_mut() {
        // Update primitive to node mapping
        let start = new_node.first_index;
        let end = new_node.first_index + new_node.prim_count;
        if primitives_to_nodes.len() < end as usize {
            primitives_to_nodes.resize(end as usize, 0);
        }
        for prim_id in start..end {
            primitives_to_nodes[prim_id as usize] = new_node_id as u32;
        }
    }

    // Need to work up the tree updating the aabbs since we just added a node.
    bvh.refit_from_fast(new_parent_id, &parents);

    new_node_id
}

/// Slow at building, makes a slow bvh, just for testing insertion.
/// Can result in very deep BVHs in some cases.
/// Consider using `bvh.max_depth = Some(bvh.depth(0).max(DEFAULT_MAX_STACK_DEPTH));`
/// (which shouldn't typically be needed, even in huge scenes)
///
/// Dramatically slower than ploc at both building and traversal. Easily 10x or 100x slower at building.
/// (goes up by something like n^3 after a certain threshold).
/// (BVH quality still improved afterward lot by reinsertion/collapse).
pub fn build_bvh2_by_insertion<T: Boundable>(primitives: &[T]) -> Bvh2 {
    let mut bvh = Bvh2 {
        nodes: vec![Bvh2Node {
            aabb: primitives[0].aabb(),
            prim_count: 1,
            first_index: 0,
        }],
        primitive_indices: (0..primitives.len() as u32).collect(),
        ..Default::default()
    };

    let mut stack = HeapStack::new_with_capacity(1000);

    let mut primitives_to_nodes = vec![INVALID; primitives.len()];

    let mut parents = bvh.compute_parents();
    for prim_id in 1..primitives.len() {
        insert_leaf_node(
            &mut bvh,
            Bvh2Node {
                aabb: primitives[prim_id].aabb(),
                prim_count: 1,
                first_index: prim_id as u32,
            },
            &mut parents,
            &mut stack,
            Some(&mut primitives_to_nodes),
        );
    }

    #[cfg(debug_assertions)]
    {
        bvh.validate_primitives_to_nodes(&primitives_to_nodes);
        bvh.validate(primitives, false, false);
        bvh.validate_parents(&parents);
    }

    bvh
}

/// Just here to for testing/benchmarking/validating leaf removed and inserting. See reinsertion.rs if you want to
/// optimize a BVH2. This currently actually tends to make a good bvh slower since doing a lot of insert_leaf_node tends
/// to result in very deep BVHs.
pub fn slow_leaf_reinsertion(
    bvh: &mut Bvh2,
    parents: &mut Vec<u32>,
    mut primitives_to_nodes: Option<&mut Vec<u32>>,
) {
    let mut stack = HeapStack::new_with_capacity(1000);
    for node_id in 1..bvh.nodes.len() {
        if bvh.nodes.len() <= node_id {
            break;
        }
        if bvh.nodes[node_id].is_leaf() {
            // Assert that the parent of this node is not a leaf (a parent could never be a leaf)
            assert!(!bvh.nodes[parents[node_id] as usize].is_leaf());
            // If the node is a leaf, remove it
            let removed_leaf =
                remove_leaf(bvh, node_id, parents, primitives_to_nodes.as_deref_mut());
            // Insert it again, maybe it will find a better spot
            insert_leaf_node(
                bvh,
                removed_leaf,
                parents,
                &mut stack,
                primitives_to_nodes.as_deref_mut(),
            );
        }
    }
    #[cfg(debug_assertions)]
    {
        bvh.validate_parents(&parents);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{bvh2::builder::build_bvh2, test_util::geometry::demoscene, BvhBuildParams};

    #[test]
    fn build_by_insertion() {
        for res in 30..=32 {
            let tris = demoscene(res, 0);
            let bvh = build_bvh2_by_insertion(&tris);
            bvh.validate(&tris, false, false);
        }
    }

    #[test]
    fn test_slow_leaf_reinsertion() {
        for res in 30..=32 {
            let tris = demoscene(res, 0);

            let mut bvh = build_bvh2(
                &tris,
                BvhBuildParams::fastest_build(),
                &mut Duration::default(),
            );
            let mut primitives_to_nodes = bvh.compute_primitives_to_nodes(tris.len());
            bvh.validate(&tris, false, false);
            let mut parents = bvh.compute_parents();
            slow_leaf_reinsertion(&mut bvh, &mut parents, Some(&mut primitives_to_nodes));
            bvh.validate(&tris, false, false);
            bvh.validate_parents(&parents);
            bvh.validate_primitives_to_nodes(&primitives_to_nodes);
            bvh.reorder_in_stack_traversal_order();
            bvh.validate(&tris, false, false);
        }
    }

    #[test]
    fn remove_all_primitives() {
        let tris = demoscene(16, 0);
        let mut bvh = build_bvh2(
            &tris,
            BvhBuildParams::fastest_build(),
            &mut Duration::default(),
        );
        let mut parents = bvh.compute_parents();
        let mut primitives_to_nodes = bvh.compute_primitives_to_nodes(tris.len());

        bvh.validate_parents(&parents);
        bvh.validate_primitives_to_nodes(&primitives_to_nodes);
        bvh.validate(&tris, true, false);

        for prim_id in 0..tris.len() {
            let node_id = primitives_to_nodes[prim_id];
            let _removed_node = remove_leaf(
                &mut bvh,
                node_id as usize,
                &mut parents,
                Some(&mut primitives_to_nodes),
            );
            bvh.validate_parents(&parents);
            bvh.validate_primitives_to_nodes(&primitives_to_nodes);
            bvh.validate(&tris, true, false);
        }

        bvh.validate_parents(&parents);
        bvh.validate_primitives_to_nodes(&primitives_to_nodes);
        bvh.validate(&tris, true, false);
    }
}
