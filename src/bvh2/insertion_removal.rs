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
        let sibling = self.nodes[sibling_id];
        self.nodes[parent_id] = sibling;
        // Tell the children (or primitives) of the moved sibling where it went.
        self.relink_node(&sibling, parent_id);
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
            self.nodes[dst_right_id] = right_src_sibling;
            // Tell the children (or primitives) of the moved right_src_sibling where it went.
            self.relink_node(&right_src_sibling, dst_right_id);

            let left_src_sibling = self.nodes.pop().unwrap(); // Last node is left src sibling
            self.nodes[dst_left_id] = left_src_sibling;
            // Tell the children (or primitives) of the moved left_src_sibling where it went.
            self.relink_node(&left_src_sibling, dst_left_id);

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
    ///
    /// See "Branch and Bound" <https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf> and
    /// Jiˇrí Bittner et al. 2012 Fast Insertion-Based Optimization of Bounding Volume Hierarchies
    ///
    /// See [`Bvh2::insert_leaf_greedy()`] for a faster version that descends a single path,
    /// and that rotates on the refit walk to keep the BVH from degenerating.
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

        self.nodes.push(best_sibling_candidate);
        let new_node_id = self.nodes.len();
        self.nodes.push(new_node); // Put the new node at the very end.
        self.parents.push(new_parent_id as u32);
        self.parents.push(new_parent_id as u32);

        // Tell the children (or primitives) of the moved sibling where it went.
        self.relink_node(&best_sibling_candidate, new_sibling_id as usize);

        if !best_sibling_candidate.is_leaf() {
            // The sibling was moved to the end of the node list, but its children stayed where they were,
            // so they now come before their parent.
            self.children_are_ordered_after_parents = false;
        }

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

    /// Searches the tree for the best sibling for a leaf with the given `aabb` using a greedy top-down descent.
    /// The best sibling is classified as the sibling that if chosen it would increase the surface area of the BVH the least.
    ///
    /// Unlike the branch and bound search in [`Bvh2::insert_leaf()`], this only ever descends a single path from the root,
    /// always taking the child with the lower bound on what inserting beneath it could cost, and stopping as soon as
    /// neither child's bound can beat the best candidate already found. This is O(depth), needs no traversal stack,
    /// and doesn't blow up the same way as the BVH gets deeper. However, it tends to find a slightly worse sibling
    /// than the branch and bound search.
    ///
    /// # Returns
    /// The index of the best sibling found, and the depth it was found at (the root has a depth of 0).
    ///
    /// # Arguments
    /// * `aabb` - The aabb of the leaf that is going to be inserted.
    pub fn find_sibling_greedy(&self, aabb: &Aabb) -> (usize, usize) {
        // This is based on `b2FindBestSibling` in Box2D by Erin Catto.

        let center = aabb.center();
        let area = aabb.half_area();

        let root_aabb = self.nodes[0].aabb();

        // Area of the node currently being descended into.
        let mut area_base = root_aabb.half_area();

        // Area of that node once it has been inflated to also contain the new leaf.
        let mut direct_cost = root_aabb.union(aabb).half_area();

        // How much every node from the root down to the current node would have to grow.
        let mut inherited_cost = 0.0;

        let mut best_cost = direct_cost;
        let mut best_sibling_candidate_id = 0;
        let mut best_sibling_candidate_depth = 0;

        let mut current_node_index = 0;
        let mut depth = 0;

        // Descend the tree from the root, following a single greedy path.
        while !self.nodes[current_node_index].is_leaf() {
            let left_id = self.nodes[current_node_index].first_index as usize;
            let right_id = left_id + 1;

            // Cost of creating a new parent for this node and the new leaf.
            let total_cost = direct_cost + inherited_cost;
            if total_cost < best_cost {
                best_cost = total_cost;
                best_sibling_candidate_id = current_node_index;
                best_sibling_candidate_depth = depth;
            }

            inherited_cost += direct_cost - area_base;

            let left_is_leaf = self.nodes[left_id].is_leaf();
            let right_is_leaf = self.nodes[right_id].is_leaf();
            let left_aabb = *self.nodes[left_id].aabb();
            let right_aabb = *self.nodes[right_id].aabb();
            let left_direct_cost = left_aabb.union(aabb).half_area();
            let right_direct_cost = right_aabb.union(aabb).half_area();

            // Lower bound on the cost of inserting anywhere below each child.
            // Left at f32::MAX for leaves since there is nothing below them to descend into.
            let mut left_lower_cost = f32::MAX;
            let mut right_lower_cost = f32::MAX;
            let mut left_area = 0.0;
            let mut right_area = 0.0;

            if left_is_leaf {
                let cost = left_direct_cost + inherited_cost;
                if cost < best_cost {
                    best_cost = cost;
                    best_sibling_candidate_id = left_id;
                    best_sibling_candidate_depth = depth + 1;
                }
            } else {
                left_area = left_aabb.half_area();
                left_lower_cost = inherited_cost + left_direct_cost + (area - left_area).min(0.0);
            }

            if right_is_leaf {
                let cost = right_direct_cost + inherited_cost;
                if cost < best_cost {
                    best_cost = cost;
                    best_sibling_candidate_id = right_id;
                    best_sibling_candidate_depth = depth + 1;
                }
            } else {
                right_area = right_aabb.half_area();
                right_lower_cost =
                    inherited_cost + right_direct_cost + (area - right_area).min(0.0);
            }

            if best_cost <= left_lower_cost && best_cost <= right_lower_cost {
                // Neither subtree can beat the best candidate already found.
                break;
            }

            if left_lower_cost == right_lower_cost && !left_is_leaf {
                // The bounds give no clear choice, which happens when both children
                // fully contain `aabb`. Fall back to the child whose center is closest.
                left_lower_cost = (left_aabb.center() - center).length_squared();
                right_lower_cost = (right_aabb.center() - center).length_squared();
            }

            // Descend into the more promising child.
            if left_lower_cost < right_lower_cost && !left_is_leaf {
                current_node_index = left_id;
                area_base = left_area;
                direct_cost = left_direct_cost;
            } else {
                current_node_index = right_id;
                area_base = right_area;
                direct_cost = right_direct_cost;
            }

            depth += 1;
        }

        (best_sibling_candidate_id, best_sibling_candidate_depth)
    }

    /// Attaches `new_node` to the BVH as the new sibling of the node at `sibling_id`.
    ///
    /// The new parent takes the sibling's slot, so nothing above `sibling_id` moves.
    /// The displaced sibling and `new_node` are put in the adjacent pair of slots
    /// at `left_id` and `left_id + 1`. Both of those slots have to already exist
    /// and must not be reachable from the root.
    ///
    /// # Returns
    /// The index of the newly attached node (always `left_id + 1`).
    fn attach_leaf(&mut self, new_node: Bvh2Node, sibling_id: usize, left_id: usize) -> usize {
        debug_assert!(Bvh2Node::is_left_sibling(left_id));

        let new_node_id = left_id + 1;
        let sibling = self.nodes[sibling_id];

        // New parent goes in the sibling's position.
        self.nodes[sibling_id] =
            Bvh2Node::new(new_node.aabb().union(sibling.aabb()), 0, left_id as u32);
        self.nodes[left_id] = sibling;
        self.nodes[new_node_id] = new_node;
        self.parents[left_id] = sibling_id as u32;
        self.parents[new_node_id] = sibling_id as u32;

        // Tell the children (or primitives) of the moved sibling where it went.
        self.relink(left_id);

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

        if !sibling.is_leaf() || left_id < sibling_id {
            // The sibling was moved, but its children stayed where they were,
            // so they may now come before their parent. The new pair may also
            // come before the new parent when reusing freed slots.
            self.children_are_ordered_after_parents = false;
        }

        new_node_id
    }

    /// Detaches the leaf at `node_id` without compacting `Bvh2::nodes`.
    ///
    /// The sibling takes the parent's slot, and the slots at `node_id` and its sibling then fall free.
    /// They are unreachable from the root, so nothing can select one of them as an insertion sibling.
    ///
    /// Doesn't update the `primitive_indices` mapping, and leaves the `primitives_to_nodes` entries of the detached leaf
    /// pointing at `node_id`. Both are still valid if the leaf is immediately reattached, see [`Bvh2::move_leaf()`].
    ///
    /// # Returns
    /// The index of the left slot of the freed pair.
    fn detach_leaf(&mut self, node_id: usize) -> usize {
        let sibling_id = Bvh2Node::get_sibling_id(node_id);
        debug_assert_eq!(
            self.parents[node_id], self.parents[sibling_id],
            "Both children should already have the same parent."
        );
        let parent_id = self.parents[node_id] as usize;

        // Put sibling in parent's place (parent doesn't exist anymore)
        self.nodes[parent_id] = self.nodes[sibling_id];
        self.relink(parent_id);

        // Need to work up the tree updating the aabbs since we just removed a node.
        self.refit_from_fast(parent_id);

        self.children_are_ordered_after_parents = false;

        Bvh2Node::get_left_sibling_id(node_id)
    }

    /// Moves the leaf specified by `node_id` to a new position in the BVH, resizing it to `aabb`.
    /// This is the fused equivalent of [`Bvh2::remove_leaf()`] followed by [`Bvh2::insert_leaf_greedy()`].
    ///
    /// `Bvh2::nodes.len()` is unchanged across the call and `Bvh2::primitive_indices` is untouched.
    ///
    /// # Returns
    /// The index of the moved node.
    ///
    /// # Arguments
    /// * `node_id` - The index into `self.nodes` of the leaf being moved.
    /// * `aabb` - The new aabb of the leaf.
    /// * `should_rotate` - Rotate on the refit walk, which keeps the BVH more balanced
    ///   and can improve traversal speed at the cost of a slightly more expensive refit.
    pub fn move_leaf(&mut self, node_id: usize, aabb: Aabb, should_rotate: bool) -> usize {
        assert!(
            !self.uses_spatial_splits,
            "Moving leaves while using spatial splits is currently unsupported as it would require a mapping \
from one primitive to multiple nodes in `Bvh2::primitives_to_nodes`."
        );

        let mut node = self.nodes[node_id];
        assert!(node.is_leaf());
        node.set_aabb(aabb);

        if self.nodes.len() == 1 {
            self.nodes[0] = node;
            return 0;
        }
        debug_assert_ne!(node_id, 0);

        self.init_parents_if_uninit();

        let left_id = self.detach_leaf(node_id);
        let (sibling_id, sibling_depth) = self.find_sibling_greedy(node.aabb());
        let new_node_id = self.attach_leaf(node, sibling_id, left_id);
        self.update_max_depth_for_greedy_insertion(sibling_depth, should_rotate);

        // Need to work up the tree updating the aabbs since we just added a node.
        if should_rotate {
            // A rotation along the way can move the node we just attached, so we track it.
            self.refit_and_rotate_from_tracking(sibling_id, new_node_id)
        } else {
            self.refit_from_fast(sibling_id);
            new_node_id
        }
    }

    /// Moves the leaf that contains the given primitive to a new position in the BVH, resizing it to `aabb`.
    /// This is the fused equivalent of [`Bvh2::remove_primitive()`] followed by [`Bvh2::insert_primitive_greedy()`].
    /// The whole leaf is moved, so this requires that the leaf contains only this primitive.
    ///
    /// # Returns
    /// The new index of the node of this primitive.
    ///
    /// # Arguments
    /// * `aabb` - The new aabb of the primitive.
    /// * `primitive_id` - The index of the primitive being moved.
    /// * `should_rotate` - Rotate on the refit walk, which keeps the BVH more balanced
    ///   and can improve traversal speed at the cost of a slightly more expensive refit.
    pub fn move_primitive(&mut self, aabb: Aabb, primitive_id: u32, should_rotate: bool) -> usize {
        self.init_primitives_to_nodes_if_uninit();
        self.init_parents_if_uninit();
        let node_id = self.primitives_to_nodes[primitive_id as usize] as usize;
        assert_eq!(
            self.nodes[node_id].prim_count, 1,
            "Bvh2::move_primitive() would move the other primitives in this leaf along with it."
        );
        let new_node_id = self.move_leaf(node_id, aabb, should_rotate);
        debug_assert_eq!(
            self.primitives_to_nodes[primitive_id as usize],
            new_node_id as u32
        );
        new_node_id
    }

    /// Searches the tree with the greedy descent in [`Bvh2::find_sibling_greedy()`] to find a sibling
    /// for the node being inserted, then attaches it there and refits back up to the root.
    ///
    /// This is the greedy counterpart of [`Bvh2::insert_leaf()`]. It doesn't need a traversal stack
    /// and it keeps [`Bvh2::max_depth`] up to date, at the cost of sometimes picking a slightly worse sibling.
    ///
    /// # Returns
    /// The index of the newly added node. Note that with `should_rotate` this is generally not
    /// `self.nodes.len() - 1`, since a rotation can move the node that was just attached.
    ///
    /// # Arguments
    /// * `new_node` - This node must be a leaf and already have a valid `first_index` into `primitive_indices`.
    /// * `should_rotate` - Rotate on the refit walk, which keeps the BVH more balanced
    ///   and can improve traversal speed at the cost of a slightly more expensive refit.
    pub fn insert_leaf_greedy(&mut self, new_node: Bvh2Node, should_rotate: bool) -> usize {
        assert!(new_node.is_leaf());

        if self.nodes.is_empty() {
            self.nodes.push(new_node);
            self.parents.clear();
            self.parents.push(0);
            return 0;
        }

        self.init_parents_if_uninit();

        let (sibling_id, sibling_depth) = self.find_sibling_greedy(new_node.aabb());

        // Grow by an adjacent pair at the end for the displaced sibling and the new node.
        // The node count is always odd, so the first of the two is always a left sibling.
        let left_id = self.nodes.len();
        self.nodes.push(Default::default());
        self.nodes.push(Default::default());
        self.parents.push(0);
        self.parents.push(0);

        let new_node_id = self.attach_leaf(new_node, sibling_id, left_id);
        self.update_max_depth_for_greedy_insertion(sibling_depth, should_rotate);

        // Need to work up the tree updating the aabbs since we just added a node.
        if should_rotate {
            // A rotation along the way can move the node we just attached, so we track it.
            self.refit_and_rotate_from_tracking(sibling_id, new_node_id)
        } else {
            self.refit_from_fast(sibling_id);
            new_node_id
        }
    }

    /// Searches the tree with the greedy descent in [`Bvh2::find_sibling_greedy()`] to find a sibling
    /// for the primitive being inserted, then attaches it there and refits back up to the root.
    /// Updates [`Bvh2::primitive_indices`] and [`Bvh2::primitive_indices_freelist`].
    ///
    /// This is the greedy counterpart of [`Bvh2::insert_primitive()`]. It doesn't need a traversal stack
    /// and it keeps [`Bvh2::max_depth`] up to date, at the cost of sometimes picking a slightly worse sibling.
    ///
    /// # Returns
    /// The index of the newly added node.
    ///
    /// # Arguments
    /// * `aabb` - The aabb of the primitive being inserted.
    /// * `primitive_id` - The index of the primitive being inserted.
    /// * `should_rotate` - Rotate on the refit walk, which keeps the BVH more balanced
    ///   and can improve traversal speed at the cost of a slightly more expensive refit.
    pub fn insert_primitive_greedy(
        &mut self,
        aabb: Aabb,
        primitive_id: u32,
        should_rotate: bool,
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
        let new_node_id =
            self.insert_leaf_greedy(Bvh2Node::new(aabb, 1, first_index), should_rotate);
        self.primitives_to_nodes[primitive_id as usize] = new_node_id as u32;
        new_node_id
    }

    /// Grows [`Bvh2::max_depth`] to account for a leaf that was just inserted below a sibling
    /// at `sibling_depth`, where the root has a depth of 0.
    ///
    /// Note this is an upper bound estimate, not a strict bound. If the sibling is an inner node,
    /// its whole subtree is pushed one level deeper too, and the greedy descent never visits that subtree,
    /// so its height isn't known here.
    #[inline]
    fn update_max_depth_for_greedy_insertion(&mut self, sibling_depth: usize, should_rotate: bool) {
        // + 1 for the new leaf below the sibling
        // + 1 because `max_depth` is a traversal stack size rather than a node depth
        // + 1 more for a rotation that may push the new leaf one level deeper
        self.max_depth = self
            .max_depth
            .max(sibling_depth + 2 + should_rotate as usize);
    }

    /// Considers swapping one of the children of the node at `node_id` with one of the other child's children,
    /// applying the cheapest of the four such rotations if it improves the total cost of the two inner nodes below `node_id`.
    ///
    /// This incrementally recovers the quality of the BVH as leaves are inserted or the tree is refit,
    /// and helps avoid degenerating depth.
    ///
    /// # Returns
    /// The two slots whose contents were swapped, if a rotation was applied. Any node id held by the caller
    /// that is one of the two needs to be updated to the other.
    ///
    /// # Arguments
    /// * `node_id` - The index into `self.nodes` of the node to rotate around. Does nothing for leaves,
    ///   or for nodes whose children are both leaves.
    pub fn rotate_node(&mut self, node_id: usize) -> Option<(usize, usize)> {
        // This is based on `b2RotateNodes` in Box2D by Erin Catto.

        let node = self.nodes[node_id];
        if node.is_leaf() {
            return None;
        }

        // Initial subtree:
        //
        //     node
        //    /    \
        //   b      c
        //  / \    / \
        // d   e  f   g
        let b_id = node.first_index as usize;
        let c_id = b_id + 1;
        let b = self.nodes[b_id];
        let c = self.nodes[c_id];

        let b_area = b.aabb().half_area();
        let c_area = c.aabb().half_area();
        let mut best_cost = b_area + c_area;

        // The two slots to swap, the node that ends up with a different pair of children, and the aabb it ends up with.
        let mut best_rotation: Option<(usize, usize, usize, Aabb)> = None;

        if !c.is_leaf() {
            let f_id = c.first_index as usize;
            let g_id = f_id + 1;
            let f_aabb = *self.nodes[f_id].aabb();
            let g_aabb = *self.nodes[g_id].aabb();

            // Cost of swapping b and f, which leaves c covering b and g.
            //
            //       node
            //      /    \
            //     f      c
            //           / \
            //          b   g
            //         / \
            //        d   e
            let aabb_bg = b.aabb().union(&g_aabb);
            let cost_bf = b_area + aabb_bg.half_area();
            if cost_bf < best_cost {
                best_cost = cost_bf;
                best_rotation = Some((b_id, f_id, c_id, aabb_bg));
            }

            // Cost of swapping b and g, which leaves c covering b and f.
            //
            //       node
            //      /    \
            //     g      c
            //           / \
            //          f   b
            //             / \
            //            d   e
            let aabb_bf = b.aabb().union(&f_aabb);
            let cost_bg = b_area + aabb_bf.half_area();
            if cost_bg < best_cost {
                best_cost = cost_bg;
                best_rotation = Some((b_id, g_id, c_id, aabb_bf));
            }
        }

        if !b.is_leaf() {
            let d_id = b.first_index as usize;
            let e_id = d_id + 1;
            let d_aabb = *self.nodes[d_id].aabb();
            let e_aabb = *self.nodes[e_id].aabb();

            // Cost of swapping c and d, which leaves b covering c and e.
            //
            //       node
            //      /    \
            //     b      d
            //    / \
            //   c   e
            //  / \
            // f   g
            let aabb_ce = c.aabb().union(&e_aabb);
            let cost_cd = c_area + aabb_ce.half_area();
            if cost_cd < best_cost {
                best_cost = cost_cd;
                best_rotation = Some((c_id, d_id, b_id, aabb_ce));
            }

            // Cost of swapping c and e, which leaves b covering c and d.
            //
            //       node
            //      /    \
            //     b      e
            //    / \
            //   d   c
            //      / \
            //     f   g
            let aabb_cd = c.aabb().union(&d_aabb);
            let cost_ce = c_area + aabb_cd.half_area();
            if cost_ce < best_cost {
                // Last candidate, so best_cost doesn't need to be updated.
                best_rotation = Some((c_id, e_id, b_id, aabb_cd));
            }
        }

        let Some((from, to, refit_id, refit_aabb)) = best_rotation else {
            // No rotation improves the cost.
            return None;
        };

        // Box2D reassigns child pointers, but obvhs stores children as an adjacent pair.
        // Swapping the contents of the two slots has the same structural effect.
        self.nodes.swap(from, to);
        self.relink(from);
        self.relink(to);

        // Only the node that took on a different pair of children needs its aabb recomputed.
        self.nodes[refit_id].set_aabb(refit_aabb);

        self.children_are_ordered_after_parents = false;

        Some((from, to))
    }

    /// Refit the BVH working up the tree from this node, ignoring leaves,
    /// performing rotations at each node along the way where beneficial.
    ///
    /// This can only be used to refit when a single node has changed or moved.
    ///
    /// See [`Bvh2::rotate_node()`] and [`Bvh2::refit_from()`].
    pub fn refit_and_rotate_from(&mut self, index: usize) {
        self.refit_and_rotate_from_tracking(index, INVALID as usize);
    }

    /// Same as [`Bvh2::refit_and_rotate_from()`], but follows the node at `tracked_node_id`
    /// through any rotations applied along the way.
    ///
    /// # Returns
    /// The index that the tracked node ended up at.
    fn refit_and_rotate_from_tracking(
        &mut self,
        mut index: usize,
        mut tracked_node_id: usize,
    ) -> usize {
        self.init_parents_if_uninit();
        loop {
            let node = self.nodes[index];
            if !node.is_leaf() {
                let first_child_bbox = *self.nodes[node.first_index as usize].aabb();
                let second_child_bbox = *self.nodes[node.first_index as usize + 1].aabb();
                self.nodes[index].set_aabb(first_child_bbox.union(&second_child_bbox));

                // A rotation only swaps nodes below `index`, so the computed aabb stays valid.
                if let Some((from, to)) = self.rotate_node(index) {
                    if tracked_node_id == from {
                        tracked_node_id = to;
                    } else if tracked_node_id == to {
                        tracked_node_id = from;
                    }
                }
            }
            if index == 0 {
                break;
            }
            index = self.parents[index] as usize;
        }
        tracked_node_id
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
///
/// See [`build_bvh2_by_greedy_insertion()`] which doesn't have either issue.
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

/// Same as [`build_bvh2_by_insertion()`],  but with the greedy sibling search and rotations,
/// just for testing insertion.
///
/// Still slow at building compared to ploc, but uses a faster sibling search, and if `should_rotate` is true,
/// doesn't degenerate into a deep BVH the way [`build_bvh2_by_insertion()`] can.
#[doc(hidden)]
pub fn build_bvh2_by_greedy_insertion<T: Boundable>(primitives: &[T], should_rotate: bool) -> Bvh2 {
    let mut bvh = Bvh2::default();

    for prim_id in 0..primitives.len() {
        bvh.insert_primitive_greedy(primitives[prim_id].aabb(), prim_id as u32, should_rotate);
    }

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

    use glam::vec3a;

    use super::*;
    use crate::{BvhBuildParams, bvh2::builder::build_bvh2, test_util::geometry::demoscene};

    /// `count` unit boxes laid out in sorted order along a line.
    /// This is the input that degenerates a rotation-free insertion build into a list.
    fn sorted_boxes(count: usize) -> Vec<Aabb> {
        (0..count)
            .map(|i| {
                let center = vec3a(i as f32, 0.0, 0.0);
                Aabb::new(center - 0.5, center + 0.5)
            })
            .collect()
    }

    #[test]
    fn build_by_insertion() {
        for res in 30..=32 {
            let tris = demoscene(res, 0);
            let bvh = build_bvh2_by_insertion(&tris);
            bvh.validate(&tris, false, false);
        }
    }

    #[test]
    fn build_by_greedy_insertion() {
        for res in 30..=32 {
            let tris = demoscene(res, 0);
            let bvh = build_bvh2_by_greedy_insertion(&tris, true);
            bvh.validate(&tris, false, true);
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

    #[test]
    fn insert_leaf_clears_children_are_ordered_after_parents() {
        // Regression test
        //
        // If `Bvh2::children_are_ordered_after_parents` isn't cleared when inserting primitives,
        // `Bvh2::refit_all()` takes the fast path, and computes some parents from children
        // it hasn't updated yet, leaving nodes that don't fit their children.

        let tris = demoscene(16, 0);

        let mut bvh = build_bvh2(
            &tris,
            BvhBuildParams::medium_build(),
            &mut Duration::default(),
        );
        bvh.init_primitives_to_nodes_if_uninit();
        bvh.init_parents_if_uninit();

        // Remove and re-insert some primitives.
        let mut stack = HeapStack::new_with_capacity(1000);
        let primitive_ids = (0..tris.len() as u32).step_by(32).collect::<Vec<_>>();

        for &primitive_id in &primitive_ids {
            bvh.remove_primitive(primitive_id);
        }

        // Removal clears `children_are_ordered_after_parents`, so we restore the ordering
        // here so that we can test that insertion clears it again.
        bvh.reorder_in_stack_traversal_order();
        assert!(bvh.children_are_ordered_after_parents);

        for &primitive_id in &primitive_ids {
            let mut aabb = tris[primitive_id as usize].aabb();
            aabb.min -= 0.05;
            aabb.max += 0.05;
            bvh.insert_primitive(aabb, primitive_id, &mut stack);
        }

        bvh.max_depth = (bvh.depth(0) + 1).max(DEFAULT_MAX_STACK_DEPTH);

        bvh.validate(&tris, false, false);

        // Grow every leaf, then refit the whole bvh.
        for node in &mut bvh.nodes {
            if node.is_leaf() {
                let mut aabb = *node.aabb();
                aabb.max += 0.1;
                node.set_aabb(aabb);
            }
        }
        bvh.refit_all();

        // Every inner node should tightly fit its children.
        for (node_id, node) in bvh.nodes.iter().enumerate() {
            if !node.is_leaf() {
                let children_aabb = bvh.nodes[node.first_index as usize]
                    .aabb()
                    .union(bvh.nodes[node.first_index as usize + 1].aabb());
                assert_eq!(
                    *node.aabb(),
                    children_aabb,
                    "node {node_id} does not fit its children after refit_all()"
                );
            }
        }
    }

    #[test]
    fn greedy_insertion_maintains_max_depth() {
        // The greedy descent knows the depth of the sibling it picked, so insertion can keep `max_depth` up to date.
        // Bvh2::validate() requires it to be a valid stack size, so a stale `max_depth` would fail the validation below.
        let tris = demoscene(32, 0);
        let bvh = build_bvh2_by_greedy_insertion(&tris, true);

        assert!(
            bvh.depth(0) <= bvh.max_depth,
            "max_depth ({}) does not cover the actual bvh depth ({})",
            bvh.max_depth,
            bvh.depth(0)
        );
        bvh.validate(&tris, false, true);
    }

    #[test]
    fn greedy_insertion_without_rotations() {
        let tris = demoscene(24, 0);

        let mut bvh = Bvh2::default();
        for (prim_id, tri) in tris.iter().enumerate() {
            let node_id = bvh.insert_primitive_greedy(tri.aabb(), prim_id as u32, false);
            if prim_id > 0 {
                assert_eq!(node_id, bvh.nodes.len() - 1);
            }
            assert_eq!(bvh.primitives_to_nodes[prim_id], node_id as u32);
        }
        bvh.validate(&tris, false, true);

        // Moving without rotations reuses the freed pair, so the node count still doesn't change.
        let node_count = bvh.nodes.len();
        for (prim_id, tri) in tris.iter().enumerate() {
            let mut aabb = tri.aabb();
            aabb.min -= 0.05;
            aabb.max += 0.05;
            let node_id = bvh.move_primitive(aabb, prim_id as u32, false);
            assert_eq!(bvh.nodes.len(), node_count);
            assert_eq!(bvh.primitives_to_nodes[prim_id], node_id as u32);
        }
        bvh.validate_parents();
        bvh.validate_primitives_to_nodes();

        // The rotation-free tree should be deeper than the rotated one for the same input.
        let rotated = build_bvh2_by_greedy_insertion(&tris, true);
        let unrotated = build_bvh2_by_greedy_insertion(&tris, false);
        assert!(
            rotated.depth(0) < unrotated.depth(0),
            "rotations should produce a shallower bvh: {} vs {}",
            rotated.depth(0),
            unrotated.depth(0)
        );

        // `max_depth` should stay a valid stack size either way.
        assert!(rotated.depth(0) <= rotated.max_depth);
        assert!(unrotated.depth(0) <= unrotated.max_depth);
    }

    #[test]
    fn rotations_prevent_degenerate_bvh() {
        // With plain insertion, sorted input degenerates the BVH into a list.
        // Greedy insertion rotates on the refit walk, which recovers the quality incrementally.
        let boxes = sorted_boxes(1024);

        let mut with_rotations = Bvh2::default();
        for (prim_id, aabb) in boxes.iter().enumerate() {
            with_rotations.insert_primitive_greedy(*aabb, prim_id as u32, true);
        }
        with_rotations.validate(&boxes, false, true);

        let mut without_rotations = Bvh2::default();
        let mut stack = HeapStack::new_with_capacity(1000);
        for (prim_id, aabb) in boxes.iter().enumerate() {
            without_rotations.insert_primitive(*aabb, prim_id as u32, &mut stack);
        }

        let rotated_depth = with_rotations.depth(0);
        let unrotated_depth = without_rotations.depth(0);

        // This is 11 vs 513 at the time of writing, but for the test,
        // pinning exact numbers probably isn't worthwile.
        assert!(
            rotated_depth * 45 < unrotated_depth,
            "rotations should keep the bvh far shallower: {rotated_depth} vs {unrotated_depth}"
        );
    }

    #[test]
    fn rotate_node_preserves_aabbs() {
        let tris = demoscene(24, 0);

        let mut bvh = build_bvh2(
            &tris,
            BvhBuildParams::fastest_build(),
            &mut Duration::default(),
        );
        bvh.init_primitives_to_nodes_if_uninit();
        bvh.init_parents_if_uninit();

        for node_id in 0..bvh.nodes.len() {
            let aabb = *bvh.nodes[node_id].aabb();
            bvh.rotate_node(node_id);
            assert_eq!(
                *bvh.nodes[node_id].aabb(),
                aabb,
                "rotating node {node_id} changed its aabb"
            );
            bvh.validate(&tris, false, true);
        }
    }

    #[test]
    fn move_all_primitives() {
        let tris = demoscene(16, 0);

        let mut bvh = build_bvh2(
            &tris,
            BvhBuildParams::fastest_build(),
            &mut Duration::default(),
        );
        bvh.init_primitives_to_nodes_if_uninit();
        bvh.init_parents_if_uninit();
        bvh.validate(&tris, false, false);

        let node_count = bvh.nodes.len();
        let primitive_indices = bvh.primitive_indices.clone();

        // Move every primitive to a grown aabb, then back to its original one.
        for primitive_id in 0..tris.len() as u32 {
            let mut aabb = tris[primitive_id as usize].aabb();
            aabb.min -= 0.05;
            aabb.max += 0.05;
            bvh.move_primitive(aabb, primitive_id, true);
            assert_eq!(bvh.nodes.len(), node_count);
            bvh.validate_parents();
            bvh.validate_primitives_to_nodes();
        }
        for primitive_id in 0..tris.len() as u32 {
            bvh.move_primitive(tris[primitive_id as usize].aabb(), primitive_id, true);
            assert_eq!(bvh.nodes.len(), node_count);
        }

        assert_eq!(bvh.primitive_indices, primitive_indices);
        bvh.validate(&tris, false, true);
    }

    #[test]
    fn move_leaf_with_single_node_bvh() {
        let tris = demoscene(16, 0);

        let mut bvh = Bvh2::default();
        bvh.insert_primitive_greedy(tris[0].aabb(), 0, true);
        assert_eq!(bvh.nodes.len(), 1);

        let mut aabb = tris[0].aabb();
        aabb.max += 1.0;
        assert_eq!(bvh.move_leaf(0, aabb, true), 0);
        assert_eq!(bvh.nodes.len(), 1);
        assert_eq!(*bvh.nodes[0].aabb(), aabb);

        bvh.insert_primitive_greedy(tris[1].aabb(), 1, true);
        bvh.move_primitive(tris[0].aabb(), 0, true);
        bvh.validate(&tris[0..2], false, true);
    }
}
