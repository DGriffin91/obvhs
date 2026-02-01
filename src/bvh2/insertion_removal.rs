use core::f32;

use crate::{
    Boundable, INVALID,
    aabb::Aabb,
    bvh2::{Bvh2, Bvh2Node, update_primitives_to_nodes_for_node},
    faststack::{FastStack, HeapStack},
};

use super::DEFAULT_MAX_STACK_DEPTH;

// The index and inherited_cost of a given candidate sibling used for insertion.
#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SiblingInsertionCandidate {
    inherited_cost: f32,
    index: u32,
}

impl Bvh2 {
    /// Removes and returns the leaf specified by `node_id`.
    /// Puts `node_id` sibling in its parents place then moves the last two nodes into the now empty slots at `node_id`
    /// and its sibling.
    ///
    /// Doesn't update the primitive_indices mapping. If this node is just going to be re-inserted again, nothing needs
    /// to be done with primitive_indices, the mapping will still be valid. If this primitive needs to be removed
    /// permanently see Bvh2::remove_primitive().
    ///
    /// # Arguments
    /// * `node_id` - The index into self.nodes of the node that is to be removed
    pub fn remove_leaf(&mut self, node_id: usize) -> Bvh2Node {
        assert!(
            !self.uses_spatial_splits,
            "Removing leaves while using spatial splits is currently unsupported as it would require a mapping \
from one primitive to multiple nodes in `Bvh2::primitives_to_nodes`."
        );

        let node_to_remove = self.nodes[node_id];
        assert!(node_to_remove.is_leaf());

        if self.nodes.len() == 1 {
            // Special case if the BVH is just a leaf
            self.nodes.clear();
            self.parents.clear();
            self.primitives_to_nodes.clear();
            return node_to_remove;
        }

        // if primitives_to_nodes has already been initialized
        if !self.primitives_to_nodes.is_empty() {
            // Invalidate primitives_to_nodes instances
            for node_prim_id in
                node_to_remove.first_index..node_to_remove.first_index + node_to_remove.prim_count
            {
                let direct_prim_id = self.primitive_indices[node_prim_id as usize];
                self.primitives_to_nodes[direct_prim_id as usize] = INVALID;
            }
        }

        let sibling_id = Bvh2Node::get_sibling_id(node_id);
        debug_assert_eq!(self.parents[node_id], self.parents[sibling_id]); // Both children should already have the same parent.
        let mut parent_id = self.parents[node_id] as usize;

        // Put sibling in parent's place (parent doesn't exist anymore)
        self.nodes[parent_id] = self.nodes[sibling_id];
        let sibling = &mut self.nodes[parent_id];
        if sibling.is_leaf() {
            // Tell primitives where their node went.
            update_primitives_to_nodes_for_node(
                sibling,
                parent_id,
                &self.primitive_indices,
                &mut self.primitives_to_nodes,
            )
        } else {
            // Tell children of sibling where their parent went.
            let left_sibling_child = sibling.first_index as usize;
            self.parents[left_sibling_child] = parent_id as u32;
            self.parents[left_sibling_child + 1] = parent_id as u32;
        }
        // Don't need to update other parents here since the parent that was for this `parent_id` slot is now the direct
        // parent of the moved sibling, and the parents of `node_id` and `sibling_id` are updated below.

        // Now slots at both node_id and sibling_id are empty.
        // Take the last two nodes "src" and put them in those now empty "dst" slots.
        let end_nodes = node_id >= self.nodes.len() - 2;
        if end_nodes {
            // If these were already the last 2 nodes in the list we can just discard both.
            self.nodes.pop().unwrap();
            self.nodes.pop().unwrap();
            self.parents.pop().unwrap();
            self.parents.pop().unwrap();
        } else {
            let dst_left_id = Bvh2Node::get_left_sibling_id(node_id);
            let dst_right_id = Bvh2Node::get_right_sibling_id(node_id);

            let src_left_id = self.nodes.len() as u32 - 2;
            let src_right_id = Bvh2Node::get_sibling_id32(src_left_id);
            let src_right_parent = self.parents.pop().unwrap();
            let src_left_parent = self.parents.pop().unwrap();

            self.parents[dst_left_id] = src_left_parent;
            self.parents[dst_right_id] = src_right_parent;

            debug_assert_eq!(src_left_parent, src_right_parent); // Both children should already have the same parent.
            let parent_of_relocated = &mut self.nodes[src_left_parent as usize];
            debug_assert!(!parent_of_relocated.is_leaf());
            debug_assert_eq!(parent_of_relocated.first_index, src_left_id);
            debug_assert_eq!(parent_of_relocated.first_index + 1, src_right_id);
            // Tell the actual parent of the nodes that are moving where they're going to be now.
            self.nodes[src_left_parent as usize].first_index = dst_left_id as u32;

            let right_src_sibling = self.nodes.pop().unwrap(); // Last node is right src sibling
            if right_src_sibling.is_leaf() {
                // Tell primitives where their node went.
                update_primitives_to_nodes_for_node(
                    &right_src_sibling,
                    dst_right_id,
                    &self.primitive_indices,
                    &mut self.primitives_to_nodes,
                );
            } else {
                // Go to children of right_src_sibling and tell them where their parent went
                self.parents[right_src_sibling.first_index as usize] = dst_right_id as u32;
                self.parents[right_src_sibling.first_index as usize + 1] = dst_right_id as u32;
            }
            self.nodes[dst_right_id] = right_src_sibling;

            let left_src_sibling = self.nodes.pop().unwrap(); // Last node is left src sibling
            if left_src_sibling.is_leaf() {
                // Tell primitives where their node went.
                update_primitives_to_nodes_for_node(
                    &left_src_sibling,
                    dst_left_id,
                    &self.primitive_indices,
                    &mut self.primitives_to_nodes,
                );
            } else {
                // Go to children of left_src_sibling and tell them where their parent went
                self.parents[left_src_sibling.first_index as usize] = dst_left_id as u32;
                self.parents[left_src_sibling.first_index as usize + 1] = dst_left_id as u32;
            }
            self.nodes[dst_left_id] = left_src_sibling;

            // If the to be removed node's parent was at the end of the array and has now moved update parent_id:
            if parent_id as u32 == src_left_id {
                parent_id = dst_left_id;
            }
            if parent_id as u32 == src_right_id {
                parent_id = dst_right_id;
            }
        }

        // Need to work up the tree updating the aabbs since we just removed a node.
        self.refit_from_fast(parent_id);

        self.children_are_ordered_after_parents = false;
        // Return the removed node.
        node_to_remove
    }

    /// Searches the tree recursively to find the best sibling for the node being inserted. The best sibling is
    /// classified as the sibling that if chosen it would increase the surface area of the BVH the least.
    /// When the best sibling is found, a parent of both the sibling and the new node is put in the location of
    /// the sibling and both the sibling and new node are added to the end of the bvh.nodes.
    /// See "Branch and Bound" <https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf>
    /// Jiˇrí Bittner et al. 2012 Fast Insertion-Based Optimization of Bounding Volume Hierarchies
    ///
    /// # Returns
    /// The index of the newly added node (always `bvh.nodes.len() - 1` since the node it put at the end).
    ///
    /// # Arguments
    /// * `new_node` - This node must be a leaf and already have a valid first_index into primitive_indices
    /// * `stack` - Used for the traversal stack. Needs to be large enough to initially accommodate traversal to the
    ///   deepest leaf of the BVH. insert_leaf() will resize this stack after traversal to be at least 2x the
    ///   required size. This ends up being quite a bit faster than using a Vec and works well when inserting multiple
    ///   nodes. But does require the user to provide a good initial guess. SiblingInsertionCandidate is tiny so be
    ///   generous. Something like: `stack.reserve(bvh.depth(0) * 2).max(1000);` If you are inserting a lot of leaves
    ///   don't call bvh.depth(0) with each leaf just let insert_leaf() resize the stack as needed.
    pub fn insert_leaf(
        &mut self,
        new_node: Bvh2Node,
        stack: &mut HeapStack<SiblingInsertionCandidate>,
    ) -> usize {
        assert!(new_node.is_leaf());

        if self.nodes.is_empty() {
            self.nodes.push(new_node);
            self.parents.clear();
            self.parents.push(0);
            return 0;
        }

        self.init_parents_if_uninit();

        let mut min_cost = f32::MAX;
        let mut best_sibling_candidate_id = 0;
        let mut max_stack_len = 1;
        let new_node_cost = new_node.aabb().half_area();

        stack.clear();
        let root_aabb = self.nodes[0].aabb();

        // Traverse the BVH to find the best sibling
        stack.push(SiblingInsertionCandidate {
            inherited_cost: root_aabb.union(new_node.aabb()).half_area() - root_aabb.half_area(),
            index: 0,
        });
        while let Some(sibling_candidate) = stack.pop() {
            let current_node_index = sibling_candidate.index as usize;

            let candidate = &self.nodes[current_node_index];

            let direct_cost = candidate.aabb().union(new_node.aabb()).half_area();
            let total_cost = direct_cost + sibling_candidate.inherited_cost;

            if total_cost < min_cost {
                min_cost = total_cost;
                best_sibling_candidate_id = current_node_index;
            }

            // If this is not a leaf, it's possible a better cost could be found further down.
            if !candidate.is_leaf() {
                let inherited_cost = total_cost - candidate.aabb().half_area();
                let min_subtree_cost = new_node_cost + inherited_cost;
                if min_subtree_cost < min_cost {
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

        let best_sibling_candidate = self.nodes[best_sibling_candidate_id];

        // To avoid having gaps or re-arranging the BVH:
        // The new parent goes in the sibling's position.
        // The sibling and new node go on the end.
        let new_sibling_id = self.nodes.len() as u32;
        let new_parent = Bvh2Node::new(
            new_node.aabb().union(best_sibling_candidate.aabb()),
            0,
            new_sibling_id,
        );

        // New parent goes in the sibling's position.
        let new_parent_id = best_sibling_candidate_id;
        self.nodes[new_parent_id] = new_parent;

        if best_sibling_candidate.is_leaf() {
            // Tell primitives where their node went.
            update_primitives_to_nodes_for_node(
                &best_sibling_candidate,
                new_sibling_id as usize,
                &self.primitive_indices,
                &mut self.primitives_to_nodes,
            )
        } else {
            // If the best selected sibling was an inner node, we need to update the parents mapping so that the children of
            // that node point to the new location that it's being moved to.
            self.parents[best_sibling_candidate.first_index as usize] = new_sibling_id;
            self.parents[best_sibling_candidate.first_index as usize + 1] = new_sibling_id;
        }
        self.nodes.push(best_sibling_candidate);
        let new_node_id = self.nodes.len();
        self.nodes.push(new_node); // Put the new node at the very end.
        self.parents.push(new_parent_id as u32);
        self.parents.push(new_parent_id as u32);

        // if primitives_to_nodes has already been initialized
        if !self.primitives_to_nodes.is_empty() {
            // Tell primitives where their node went.
            let end = new_node.first_index + new_node.prim_count;
            if self.primitives_to_nodes.len() < end as usize {
                // Since we are adding a primitive it's possible that primitives_to_nodes is not large enough yet.
                self.primitives_to_nodes.resize(end as usize, INVALID);
            }
            update_primitives_to_nodes_for_node(
                &new_node,
                new_node_id,
                &self.primitive_indices,
                &mut self.primitives_to_nodes,
            )
        }

        // Need to work up the tree updating the aabbs since we just added a node.
        self.refit_from_fast(new_parent_id);

        new_node_id
    }

    /// Removes the leaf that contains the given primitive. Should be correct for nodes with multiple primitives per
    /// leaf but faster for nodes with only one primitive per leaf, and will leave node aabb oversized.
    /// Updates Bvh2::primitive_indices and Bvh2::primitive_indices_freelist.
    ///
    /// # Arguments
    /// * `primitive_id` - The index of the primitive being removed.
    pub fn remove_primitive(&mut self, primitive_id: u32) {
        assert!(
            !self.uses_spatial_splits,
            "Removing primitives while using spatial splits is currently unsupported as it would require a mapping \
from one primitive to multiple nodes in `Bvh2::primitives_to_nodes`."
        );
        let remove_primitive_id = primitive_id;
        self.init_parents_if_uninit();
        self.init_primitives_to_nodes_if_uninit();

        let node_id = self.primitives_to_nodes[remove_primitive_id as usize];

        let node = &self.nodes[node_id as usize];
        assert!(node.is_leaf());
        if node.prim_count == 1 {
            let removed_leaf = self.remove_leaf(node_id as usize);
            self.primitive_indices_freelist
                .push(removed_leaf.first_index);
            self.primitive_indices[removed_leaf.first_index as usize] = INVALID;
        } else {
            // Update leaf with the remaining primitives, use the existing leftover space in primitive_indices and
            // only add the removed primitive to the freelist

            let node = &mut self.nodes[node_id as usize];

            let start = node.first_index as usize;
            let end = (node.first_index + node.prim_count) as usize;
            let last = end - 1;
            let mut spare_spot_id = start;
            // Condense primitive_indices for this node.
            for node_prim_id in start..end {
                let direct_prim_id = self.primitive_indices[node_prim_id];
                if direct_prim_id == remove_primitive_id {
                    break;
                }
                spare_spot_id += 1;
            }
            if spare_spot_id < last {
                self.primitive_indices[spare_spot_id] = self.primitive_indices[last];
            }
            // Free now open last position.
            self.primitive_indices_freelist.push(last as u32);
            self.primitive_indices[last] = INVALID;

            assert!(node.prim_count > 1);
            node.prim_count -= 1;
        }

        if self.primitives_to_nodes.len() > remove_primitive_id as usize {
            self.primitives_to_nodes[remove_primitive_id as usize] = INVALID;
        }
    }

    /// Searches the tree recursively to find the best sibling for the primitive being inserted
    /// (see Bvh2::insert_leaf()). Updates Bvh2::primitive_indices and Bvh2::primitive_indices_freelist.
    ///
    /// # Returns
    /// The index of the newly added node.
    ///
    /// # Arguments
    /// * `bvh` - The Bvh2 the new node is being added to
    /// * `primitive_id` - The index of the primitive being inserted.
    /// * `stack` - Used for the traversal stack. Needs to be large enough to initially accommodate traversal to the
    ///   deepest leaf of the BVH. insert_leaf() will resize this stack after traversal to be at least 2x the
    ///   required size. This ends up being quite a bit faster than using a Vec and works well when inserting multiple
    ///   nodes. But does require the user to provide a good initial guess. SiblingInsertionCandidate is tiny so be
    ///   generous. Something like: `stack.reserve(bvh.depth(0) * 2).max(1000);` If you are inserting a lot of leaves
    ///   don't call bvh.depth(0) with each leaf just let insert_leaf() resize the stack as needed.
    pub fn insert_primitive(
        &mut self,
        aabb: Aabb,
        primitive_id: u32,
        stack: &mut HeapStack<SiblingInsertionCandidate>,
    ) -> usize {
        self.init_primitives_to_nodes_if_uninit();
        self.init_parents_if_uninit();
        if self.primitives_to_nodes.len() <= primitive_id as usize {
            self.primitives_to_nodes
                .resize(primitive_id as usize + 1, INVALID);
        }
        let first_index = if let Some(free_slot) = self.primitive_indices_freelist.pop() {
            self.primitive_indices[free_slot as usize] = primitive_id;
            free_slot
        } else {
            self.primitive_indices.push(primitive_id);
            self.primitive_indices.len() as u32 - 1
        };
        let new_node_id = self.insert_leaf(Bvh2Node::new(aabb, 1, first_index), stack);
        self.primitives_to_nodes[primitive_id as usize] = new_node_id as u32;
        new_node_id
    }
}

/// Slow at building, makes a slow bvh, just for testing insertion.
/// Can result in very deep BVHs in some cases.
///
/// Dramatically slower than ploc at both building and traversal. Easily 10x or 100x slower at building.
/// (goes up by something like n^3 after a certain threshold).
/// (BVH quality still improved afterward lot by reinsertion/collapse).
#[doc(hidden)]
pub fn build_bvh2_by_insertion<T: Boundable>(primitives: &[T]) -> Bvh2 {
    let mut bvh = Bvh2::default();

    let mut stack = HeapStack::new_with_capacity(1000);

    for prim_id in 1..primitives.len() {
        bvh.insert_primitive(primitives[prim_id].aabb(), prim_id as u32, &mut stack);
    }

    // Update max depth for validate
    bvh.max_depth = (bvh.depth(0) + 1).max(DEFAULT_MAX_STACK_DEPTH);

    #[cfg(debug_assertions)]
    {
        bvh.validate(primitives, false, true);
    }

    bvh
}

/// Just here to for testing/benchmarking/validating leaf removed and inserting. See reinsertion.rs if you want to
/// optimize a BVH2. This currently actually tends to make a good bvh slower since doing a lot of insert_leaf_node tends
/// to result in very deep BVHs.
#[doc(hidden)]
pub fn slow_leaf_reinsertion(bvh: &mut Bvh2) {
    let mut stack = HeapStack::new_with_capacity(1000);
    for node_id in 1..bvh.nodes.len() {
        if bvh.nodes.len() <= node_id {
            break;
        }
        if bvh.nodes[node_id].is_leaf() {
            // Assert that the parent of this node is not a leaf (a parent could never be a leaf)
            assert!(!bvh.nodes[bvh.parents[node_id] as usize].is_leaf());
            // If the node is a leaf, remove it
            let removed_leaf = bvh.remove_leaf(node_id);
            // Insert it again, maybe it will find a better spot
            bvh.insert_leaf(removed_leaf, &mut stack);
        }
    }
    #[cfg(debug_assertions)]
    {
        bvh.validate_parents();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{BvhBuildParams, bvh2::builder::build_bvh2, test_util::geometry::demoscene};

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
                BvhBuildParams::medium_build(),
                &mut Duration::default(),
            );
            bvh.init_primitives_to_nodes_if_uninit();
            bvh.init_parents_if_uninit();
            slow_leaf_reinsertion(&mut bvh);
            bvh.validate(&tris, false, false);
            bvh.reorder_in_stack_traversal_order();
            bvh.validate(&tris, false, false);
        }
    }

    #[test]
    fn remove_all_primitives() {
        let tris = demoscene(16, 0);

        // Test with both a bvh that only has one primitive per leaf
        // and also with one that has multiple primitives per leaf.
        let bvh1 = build_bvh2(
            &tris,
            BvhBuildParams::fastest_build(),
            &mut Duration::default(),
        );
        let bvh2 = build_bvh2(
            &tris,
            BvhBuildParams::medium_build(),
            &mut Duration::default(),
        );

        let mut found_leaf_with_multiple_nodes = false;
        for node in &bvh2.nodes {
            if node.prim_count > 1 {
                found_leaf_with_multiple_nodes = true;
                break;
            }
        }
        if !found_leaf_with_multiple_nodes {
            panic!(
                "Test remove_all_primitives bvh2 should have some nodes that contain multiple primitives"
            );
        }

        for bvh in &mut [bvh1, bvh2] {
            bvh.init_primitives_to_nodes_if_uninit();
            bvh.init_parents_if_uninit();
            bvh.validate(&tris, false, false);

            for primitive_id in 0..tris.len() as u32 {
                bvh.remove_primitive(primitive_id);
                bvh.validate(&tris, false, false);
            }

            assert_eq!(bvh.nodes.len(), 0);
            assert_eq!(bvh.parents.len(), 0);
            assert_eq!(bvh.primitives_to_nodes.len(), 0);
            bvh.validate(&tris, false, false);
        }
    }

    #[test]
    fn remove_and_insert_all_primitives() {
        let tris = demoscene(16, 0);

        let mut bvh = build_bvh2(
            &tris,
            BvhBuildParams::medium_build(),
            &mut Duration::default(),
        );
        bvh.init_primitives_to_nodes_if_uninit();
        bvh.init_parents_if_uninit();
        bvh.validate(&tris, false, false);

        let mut stack = HeapStack::new_with_capacity(1000);

        for primitive_id in 0..tris.len() as u32 {
            bvh.remove_primitive(primitive_id);
            bvh.validate(&tris, false, false);
        }

        for primitive_id in 0..tris.len() as u32 {
            bvh.insert_primitive(tris[primitive_id as usize].aabb(), primitive_id, &mut stack);
            bvh.validate_primitives_to_nodes();
        }

        bvh.validate(&tris, false, false);
    }
}
