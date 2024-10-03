use core::f32;

use crate::{
    bvh2::{Bvh2, Bvh2Node},
    heapstack::HeapStack,
    Boundable,
};

pub fn remove_node(bvh: &mut Bvh2, node_id: usize, parents: &[u32]) {
    // TODO if this node is not a leaf, remove sub-tree?
    // Or maybe this would be in a different fn that would let you extract the subtree
    // TODO Use something like reorder_in_stack_traversal_order to clean up the bvh
    debug_assert!(node_id != 0);
    let parent_id = parents[node_id] as usize;
    let sibling_id = Bvh2Node::get_sibling_id(node_id);
    bvh.nodes[parent_id] = bvh.nodes[sibling_id];
    bvh.refit_from_fast(parent_id, &parents);
}

// See "Branch and Bound" https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf
pub fn insert_leaf_node(
    bvh: &mut Bvh2,
    node: Bvh2Node,
    parents: &mut Vec<u32>,
    stack: &mut HeapStack<u32>,
) {
    debug_assert!(node.is_leaf());
    let mut min_cost = f32::MAX;
    let mut best_sibling_candidate_id = 0;

    stack.clear();
    stack.push(0);
    while let Some(current_node_index) = stack.pop() {
        let current_node_index = *current_node_index as usize;

        let candidate = bvh.nodes[current_node_index];
        let candidate_cost = &candidate.aabb.half_area();

        let mut inherited_cost_delta = 0.0f32;

        // traversal_threshold is so we don't bother checking up the tree for really small changes in area.
        let traversal_threshold = candidate_cost * 0.01;
        let mut index = current_node_index;
        loop {
            let aabb = &bvh.nodes[index].aabb;
            let cost_delta = aabb.union(&node.aabb).half_area() - aabb.half_area();
            inherited_cost_delta += cost_delta;
            if index == 0 || inherited_cost_delta < traversal_threshold {
                break;
            }
            index = parents[index] as usize;
        }

        let join_cost = node.aabb.union(&candidate.aabb).half_area();
        let cost = join_cost + inherited_cost_delta;
        let min_sub_tree_cost = candidate_cost + inherited_cost_delta;

        // If this is not a leaf, and it's possible a better cost could be found further down
        if !candidate.is_leaf() && min_sub_tree_cost < min_cost {
            stack.push(candidate.first_index);
            stack.push(candidate.first_index + 1);
        }

        if cost < min_cost {
            min_cost = cost;
            best_sibling_candidate_id = current_node_index;
        }
    }

    let best_sibling_candidate = bvh.nodes[best_sibling_candidate_id];
    let sibling_is_inner = !best_sibling_candidate.is_leaf();

    // To avoid having gaps or re-arranging the BVH:
    // The new parent goes in the sibling's position
    // The sibling and new node go on the end.
    let new_parent = Bvh2Node {
        aabb: node.aabb.union(&best_sibling_candidate.aabb),
        prim_count: 0,
        first_index: bvh.nodes.len() as u32,
    };

    bvh.nodes[best_sibling_candidate_id] = new_parent;
    if sibling_is_inner {
        let new_sibling_id = bvh.nodes.len() as u32;
        parents[best_sibling_candidate.first_index as usize] = new_sibling_id;
        parents[best_sibling_candidate.first_index as usize + 1] = new_sibling_id;
    }
    bvh.nodes.push(best_sibling_candidate);
    bvh.nodes.push(node);
    parents.push(best_sibling_candidate_id as u32);
    parents.push(best_sibling_candidate_id as u32);
    bvh.refit_from(best_sibling_candidate_id, &parents);
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

    let mut parents = bvh.compute_parents(); // TODO update on insert
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
