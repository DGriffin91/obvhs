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

#[derive(Debug, Default, Clone, Copy)]
pub struct SiblingCandidate {
    inherited_cost: f32,
    index: u32,
}

/// Searches through tree recursively to find the best sibling for the node being inserted. The best sibling is
/// classified as the sibling that if chosen would increase the surface area of the BVH the least.
/// When the best sibling is found, a parent of both the sibling and the new node in put in the location of
/// the sibling and both the sibling and new node are added to the end of the bvh.nodes.
/// See "Branch and Bound" https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf
pub fn insert_leaf_node(
    bvh: &mut Bvh2,
    new_node: Bvh2Node,
    parents: &mut Vec<u32>,
    stack: &mut HeapStack<SiblingCandidate>,
) {
    debug_assert!(new_node.is_leaf());
    let mut min_cost = f32::MAX;
    let mut best_sibling_candidate_id = 0;

    stack.clear();

    let root_aabb = bvh.nodes[0].aabb;

    // Traverse the BVH to find the best sibling
    stack.push(SiblingCandidate {
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
                stack.push(SiblingCandidate {
                    inherited_cost,
                    index: candidate.first_index,
                });
                stack.push(SiblingCandidate {
                    inherited_cost,
                    index: candidate.first_index + 1,
                });
            }
        }
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
    bvh.nodes.push(new_node);
    parents.push(new_parent_id as u32);
    parents.push(new_parent_id as u32);
    // Need to work up the tree updating the aabbs since we just added a node.
    bvh.refit_from_fast(new_parent_id, &parents);
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
