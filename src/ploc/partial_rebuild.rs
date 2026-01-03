use std::{mem, u32};

use crate::{
    bvh2::{Bvh2, node::Bvh2Node},
    fast_stack,
    faststack::FastStack,
    ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
};

const SUBTREE_ROOT: u32 = u32::MAX;

impl PlocBuilder {
    /// Partially rebuild the bvh. The given set of leaves and the subtrees that do not include any of the given leaves
    /// will be built into a new bvh. If the set of leaves is a small enough proportion of the total this can be faster
    /// since there may be large portions of the BVH that don't need to be updated. If the proportion is too high it
    /// can be faster to build from scratch instead, avoiding the overhead of doing a partial rebuild. If only a tiny
    /// proportion are updated, it might be faster to selectively reinsert only the few leaves that need to be updated.
    pub fn partial_rebuild(
        &mut self,
        bvh: &mut Bvh2,
        temp_bvh: &mut Bvh2,
        leaves: &[u32],
        search_distance: PlocSearchDistance,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        if bvh.nodes.len() < 2 {
            return;
        }
        if leaves.is_empty() {
            return;
        }

        temp_bvh.reset_for_reuse(bvh.primitive_indices.len(), None);

        bvh.init_parents_if_uninit();
        self.current_nodes.clear();
        self.next_nodes.clear();
        self.mortons.clear();

        // Bottom up traverse flagging nodes as being parents of leaves that need to be rebuilt.
        let mut flagged: Vec<bool> = vec![false; bvh.nodes.len()];
        for leaf_id in leaves {
            let mut index = *leaf_id as usize;
            debug_assert!(bvh.nodes[index].is_leaf());
            flagged[index] = true;
            while index > 0 {
                index = bvh.parents[index] as usize;
                if flagged[index] {
                    // If already flagged don't need to continue up further, above this has already been traversed.
                    break;
                }
                flagged[index] = true;
            }
        }

        // Top down traverse to collect leaves and unflagged subtrees.
        fast_stack!(u32, (96, 192), bvh.max_depth, stack, {
            stack.push(1);
            while let Some(left_node_index) = stack.pop() {
                for node_index in [left_node_index as usize, left_node_index as usize + 1] {
                    let node = &bvh.nodes[node_index];
                    let flag = flagged[node_index];

                    if node.is_leaf() {
                        self.current_nodes.push(bvh.nodes[node_index]);
                    } else {
                        if flag {
                            stack.push(node.first_index);
                        } else {
                            // Unflagged sub tree. Make leaf node out of subtree root with index into old bvh.
                            self.current_nodes.push(Bvh2Node::new(
                                node.aabb,
                                SUBTREE_ROOT,
                                node_index as u32,
                            ));
                        }
                    }
                }
            }
        });

        // Build new BVH from collected leaves
        let total_aabb = *bvh.nodes[0].aabb();
        let sdt = search_depth_threshold;
        match search_distance {
            PlocSearchDistance::Minimum => {
                self.build_ploc_from_leaves::<1>(temp_bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::VeryLow => {
                self.build_ploc_from_leaves::<2>(temp_bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::Low => {
                self.build_ploc_from_leaves::<6>(temp_bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::Medium => {
                self.build_ploc_from_leaves::<14>(temp_bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::High => {
                self.build_ploc_from_leaves::<24>(temp_bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::VeryHigh => {
                self.build_ploc_from_leaves::<32>(temp_bvh, total_aabb, sort_precision, sdt)
            }
        }

        // Append subtrees onto new bvh
        fast_stack!((u32, u32), (96, 192), bvh.max_depth, stack, {
            for i in 0..temp_bvh.nodes.len() {
                if temp_bvh.nodes[i].prim_count == SUBTREE_ROOT {
                    let old_bvh_subtree_root = &bvh.nodes[temp_bvh.nodes[i].first_index as usize];

                    // Convert back to inner node and point to end of node list as we'll put sub tree there.
                    temp_bvh.nodes[i].prim_count = 0;
                    temp_bvh.nodes[i].first_index = temp_bvh.nodes.len() as u32;

                    stack.clear();

                    stack.push((old_bvh_subtree_root.first_index, i as u32));

                    while let Some((old_left_index, new_parent)) = stack.pop() {
                        let old_right_index = old_left_index + 1;
                        let old_left_node = &bvh.nodes[old_left_index as usize];
                        let old_right_node = &bvh.nodes[old_right_index as usize];

                        let current_left_idx = temp_bvh.nodes.len() as u32;

                        // Update parent with location of children in new bvh
                        temp_bvh.nodes[new_parent as usize].first_index = current_left_idx;

                        if !old_left_node.is_leaf() {
                            stack.push((old_left_node.first_index, temp_bvh.nodes.len() as u32));
                        }
                        temp_bvh.nodes.push(*old_left_node);

                        if !old_right_node.is_leaf() {
                            stack.push((old_right_node.first_index, temp_bvh.nodes.len() as u32));
                        }
                        temp_bvh.nodes.push(*old_right_node);
                    }
                }
            }
        });

        temp_bvh.max_depth = bvh.max_depth; //TODO should this be recalculated?

        mem::swap(&mut temp_bvh.primitive_indices, &mut bvh.primitive_indices);
        mem::swap(
            &mut temp_bvh.primitive_indices_freelist,
            &mut bvh.primitive_indices_freelist,
        );

        if !bvh.parents.is_empty() {
            temp_bvh.update_parents();
        }
        if !bvh.primitives_to_nodes.is_empty() {
            temp_bvh.update_primitives_to_nodes();
        }

        mem::swap(bvh, temp_bvh);
    }
}

#[cfg(test)]
mod tests {

    use glam::UVec2;

    use super::*;
    use crate::test_util::{geometry::demoscene, sampling::hash_noise};

    #[test]
    fn test_partial_rebuild_with_all_leaves() {
        let sm = demoscene(5, 0);
        for tris in [&demoscene(31, 0), &sm, &sm[..1], &sm[..2], &sm[..3], &[]] {
            let mut builder = PlocBuilder::with_capacity(tris.len());
            let mut temp_bvh = Bvh2::zeroed(tris.len());

            let mut bvh = builder.build(
                PlocSearchDistance::VeryLow,
                &tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);

            let leaves = bvh
                .nodes
                .iter()
                .enumerate()
                .filter(|(_i, n)| n.is_leaf())
                .map(|(i, _n)| i as u32)
                .collect::<Vec<u32>>();

            builder.partial_rebuild(
                &mut bvh,
                &mut temp_bvh,
                &leaves,
                PlocSearchDistance::VeryLow,
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);
        }
    }

    #[test]
    fn test_partial_rebuild_with_one_leaf() {
        let tris = demoscene(8, 0);

        let mut builder = PlocBuilder::with_capacity(tris.len());
        let mut temp_bvh = Bvh2::zeroed(tris.len());

        let mut bvh = builder.build(
            PlocSearchDistance::VeryLow,
            &tris,
            (0..tris.len() as u32).collect::<Vec<_>>(),
            SortPrecision::U64,
            1,
        );

        bvh.validate(&tris, false, true);

        let mut leaf = 0;

        for (i, node) in bvh.nodes.iter().enumerate() {
            if node.is_leaf() {
                leaf = i as u32;
                break;
            }
        }

        builder.partial_rebuild(
            &mut bvh,
            &mut temp_bvh,
            &[leaf],
            PlocSearchDistance::VeryLow,
            SortPrecision::U64,
            1,
        );

        bvh.validate(&tris, false, true);
    }

    #[test]
    fn test_partial_rebuild_with_random_leaves() {
        let sm = demoscene(5, 0);
        for tris in [&demoscene(31, 0), &sm, &sm[..1], &sm[..2], &sm[..3], &[]] {
            let mut builder = PlocBuilder::with_capacity(tris.len());
            let mut temp_bvh = Bvh2::zeroed(tris.len());

            let mut bvh = builder.build(
                PlocSearchDistance::VeryLow,
                &tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);

            let mut leaves = Vec::new();

            for (i, node) in bvh.nodes.iter().enumerate() {
                if node.is_leaf() && hash_noise(UVec2::ZERO, i as u32) > 0.5 {
                    leaves.push(i as u32);
                }
            }

            builder.partial_rebuild(
                &mut bvh,
                &mut temp_bvh,
                &leaves,
                PlocSearchDistance::VeryLow,
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);
        }
    }
}
