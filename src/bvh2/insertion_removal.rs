use core::f32;

use crate::{
    bvh2::{Bvh2, Bvh2Node},
    heapstack::HeapStack,
    Boundable,
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
pub fn remove_leaf(bvh: &mut Bvh2, node_id: usize, parents: &mut Vec<u32>) -> Bvh2Node {
    // TODO handle BVHs with 3 or less nodes

    let node_to_remove = bvh.nodes[node_id];
    assert!(node_to_remove.is_leaf());
    let sibling_id = Bvh2Node::get_sibling_id(node_id);
    debug_assert_eq!(parents[node_id], parents[sibling_id]); // Both children should already have the same parent.
    let mut parent_id = parents[node_id] as usize;

    // Put sibling in parent's place (parent doesn't exist anymore)
    bvh.nodes[parent_id] = bvh.nodes[sibling_id];
    if !bvh.nodes[parent_id].is_leaf() {
        // Tell children of sibling where their parent went.
        let left_sibling_child = bvh.nodes[parent_id].first_index as usize;
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
        if !right_src_sibling.is_leaf() {
            // Go to children of right_src_sibling and tell them where their parent went
            parents[right_src_sibling.first_index as usize] = dst_right_id as u32;
            parents[right_src_sibling.first_index as usize + 1] = dst_right_id as u32;
        }
        bvh.nodes[dst_right_id] = right_src_sibling;

        let left_src_sibling = bvh.nodes.pop().unwrap(); // Second to last node is left src sibling
        if !left_src_sibling.is_leaf() {
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
    bvh.refit_from(parent_id, &parents);

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
pub fn insert_leaf_node(
    bvh: &mut Bvh2,
    new_node: Bvh2Node,
    parents: &mut Vec<u32>,
    stack: &mut HeapStack<SiblingInsertionCandidate>,
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

    // If the best selected sibling was an inner node, we need to update the parents mapping so that the children of
    // that node point to the new location that it's being moved to.
    if !best_sibling_candidate.is_leaf() {
        parents[best_sibling_candidate.first_index as usize] = new_sibling_id;
        parents[best_sibling_candidate.first_index as usize + 1] = new_sibling_id;
    }
    bvh.nodes.push(best_sibling_candidate);
    let new_node_id = bvh.nodes.len();
    bvh.nodes.push(new_node); // Put the new node at the very end.
    parents.push(new_parent_id as u32);
    parents.push(new_parent_id as u32);
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
        );
    }

    bvh
}

/// Just here to for testing/benchmarking/validating leaf removed and inserting. See reinsertion.rs if you want to
/// optimize a BVH2. This currently actually tends to make a good bvh slower since doing a lot of insert_leaf_node tends
/// to result in very deep BVHs.
pub fn slow_leaf_reinsertion(bvh: &mut Bvh2) {
    let mut stack = HeapStack::new_with_capacity(1000);
    let mut parents = bvh.compute_parents();
    for node_id in 1..bvh.nodes.len() {
        if bvh.nodes.len() <= node_id {
            break;
        }
        if bvh.nodes[node_id].is_leaf() {
            // Assert that the parent of this node is not a leaf (a parent could never be a leaf)
            assert!(!bvh.nodes[parents[node_id] as usize].is_leaf());
            // If the node is a leaf, remove it
            let removed_leaf = remove_leaf(bvh, node_id, &mut parents);
            // Insert it again, maybe it will find a better spot
            insert_leaf_node(bvh, removed_leaf, &mut parents, &mut stack);
        }
    }
}
