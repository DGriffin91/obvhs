use std::{borrow::Borrow, mem, u32};

use crate::{
    bvh2::Bvh2,
    fast_stack,
    faststack::FastStack,
    ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
};

const SUBTREE_ROOT: u32 = u32::MAX;

pub fn compute_rebuild_path_flags<I, L>(bvh: &Bvh2, leaves: I, flags: &mut Vec<bool>)
where
    I: IntoIterator<Item = L>,
    L: Borrow<u32>,
{
    if bvh.nodes.len() < 2 {
        return;
    }
    if bvh.parents.is_empty() {
        panic!(
            "Parents must be init before running compute_rebuild_path_flags. Call `bvh.init_parents_if_uninit()` first."
        )
    }

    flags.clear();
    flags.resize(bvh.nodes.len(), false);
    // Bottom up traverse flagging nodes as being parents of leaves that need to be rebuilt.
    for leaf_id in leaves {
        let mut index = *leaf_id.borrow() as usize;
        debug_assert!(bvh.nodes[index].is_leaf());
        flags[index] = true;
        while index > 0 {
            index = bvh.parents[index] as usize;
            if flags[index] {
                // If already flagged don't need to continue up further, above this has already been traversed.
                break;
            }
            flags[index] = true;
        }
    }
}

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
        check_flag: impl Fn(usize) -> bool,
        search_distance: PlocSearchDistance,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        if bvh.nodes.len() < 2 {
            return;
        }

        temp_bvh.reset_for_reuse(bvh.primitive_indices.len(), None);

        self.current_nodes.clear();
        self.current_nodes.reserve(bvh.primitive_indices.len());
        self.next_nodes.clear();
        self.mortons.clear();

        // We'll temporarily use this allocation for a node slot freelist
        temp_bvh.primitive_indices.clear();

        // Top down traverse to collect leaves and unflagged subtrees.
        fast_stack!(u32, (96, 192), bvh.max_depth, stack, {
            stack.push(bvh.nodes[0].first_index);
            while let Some(left_node_index) = stack.pop() {
                for node_index in [left_node_index as usize, left_node_index as usize + 1] {
                    let node = &bvh.nodes[node_index];
                    let flag = check_flag(node_index);

                    if node.is_leaf() {
                        self.current_nodes.push(*node);
                    } else {
                        if flag {
                            stack.push(node.first_index);
                        } else {
                            // Unflagged sub tree. Make leaf node out of subtree root.
                            let mut node = *node;
                            node.prim_count = SUBTREE_ROOT;
                            self.current_nodes.push(node);
                        }
                    }
                }
                temp_bvh.primitive_indices.push(left_node_index);
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

        let freed_spots = temp_bvh.primitive_indices.len() * 2;
        if bvh.nodes.len() - freed_spots > temp_bvh.nodes.len() {
            // Weave new bvh back into old one

            assert!(!temp_bvh.nodes[0].is_leaf());
            bvh.nodes[0].aabb = temp_bvh.nodes[0].aabb;
            fast_stack!((u32, u32), (96, 192), temp_bvh.max_depth, stack, {
                stack.clear();
                stack.push((temp_bvh.nodes[0].first_index, 0));

                // Traverse new bvh, copying nodes into empty spaces in old bvh
                while let Some((temp_left_index, parent)) = stack.pop() {
                    let temp_right_index = temp_left_index + 1;

                    let mut temp_left_node = temp_bvh.nodes[temp_left_index as usize];
                    let mut temp_right_node = temp_bvh.nodes[temp_right_index as usize];

                    let new_left_slot = temp_bvh.primitive_indices.pop().unwrap() as usize;
                    let new_right_slot = new_left_slot + 1;

                    bvh.nodes[parent as usize].first_index = new_left_slot as u32;

                    if temp_left_node.prim_count == SUBTREE_ROOT {
                        temp_left_node.prim_count = 0;
                    } else if !temp_left_node.is_leaf() {
                        stack.push((temp_left_node.first_index, new_left_slot as u32));
                    }
                    if temp_right_node.prim_count == SUBTREE_ROOT {
                        temp_right_node.prim_count = 0;
                    } else if !temp_right_node.is_leaf() {
                        stack.push((temp_right_node.first_index, new_right_slot as u32));
                    }

                    bvh.nodes[new_left_slot] = temp_left_node;
                    bvh.nodes[new_right_slot] = temp_right_node;
                }
            });

            temp_bvh.primitive_indices.clear();
        } else {
            // Append old subtrees onto new bvh

            fast_stack!((u32, u32), (96, 192), bvh.max_depth, stack, {
                for i in 0..temp_bvh.nodes.len() {
                    let node = &mut temp_bvh.nodes[i];
                    if node.prim_count == SUBTREE_ROOT {
                        // Convert back to inner node.
                        node.prim_count = 0;

                        // node.first_index will point to end of node list as we'll put sub tree there but it would be
                        // overwritten later below so don't bother here: subtree_root.first_index = temp_bvh.nodes.len();

                        stack.clear();
                        stack.push((node.first_index, i as u32));

                        while let Some((old_left_index, new_parent)) = stack.pop() {
                            let old_right_index = old_left_index + 1;

                            let current_left_idx = temp_bvh.nodes.len() as u32;

                            // Update parent with location of children in new bvh
                            temp_bvh.nodes[new_parent as usize].first_index = current_left_idx;

                            let old_left_node = &bvh.nodes[old_left_index as usize];
                            let old_right_node = &bvh.nodes[old_right_index as usize];
                            if !old_left_node.is_leaf() {
                                stack.push((old_left_node.first_index, current_left_idx));
                            }
                            if !old_right_node.is_leaf() {
                                stack.push((old_right_node.first_index, current_left_idx + 1));
                            }
                            temp_bvh.nodes.push(*old_left_node);
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
            mem::swap(bvh, temp_bvh);
        }

        bvh.children_are_ordered_after_parents = false;

        if !bvh.parents.is_empty() {
            bvh.update_parents();
        }
        if !bvh.primitives_to_nodes.is_empty() {
            bvh.update_primitives_to_nodes();
        }
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
                PlocSearchDistance::Minimum,
                &tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);

            bvh.init_parents_if_uninit();
            let mut flags = Vec::new();
            compute_rebuild_path_flags(
                &bvh,
                bvh.nodes
                    .iter()
                    .enumerate()
                    .filter(|(_i, n)| n.is_leaf())
                    .map(|(i, _n)| i as u32),
                &mut flags,
            );

            builder.partial_rebuild(
                &mut bvh,
                &mut temp_bvh,
                |node_id| flags[node_id],
                PlocSearchDistance::Minimum,
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
            PlocSearchDistance::Minimum,
            &tris,
            (0..tris.len() as u32).collect::<Vec<_>>(),
            SortPrecision::U64,
            0,
        );

        bvh.validate(&tris, false, true);

        bvh.init_parents_if_uninit();
        let mut flags = Vec::new();
        compute_rebuild_path_flags(
            &bvh,
            bvh.nodes
                .iter()
                .enumerate()
                .filter(|(_i, n)| n.is_leaf())
                .map(|(i, _n)| i as u32)
                .take(1),
            &mut flags,
        );

        builder.partial_rebuild(
            &mut bvh,
            &mut temp_bvh,
            |node_id| flags[node_id],
            PlocSearchDistance::Minimum,
            SortPrecision::U64,
            0,
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
                PlocSearchDistance::Minimum,
                &tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(&tris, false, true);

            bvh.init_parents_if_uninit();
            let mut flags = Vec::new();
            compute_rebuild_path_flags(
                &bvh,
                bvh.nodes
                    .iter()
                    .enumerate()
                    .filter(|(i, n)| n.is_leaf() && hash_noise(UVec2::ZERO, *i as u32) > 0.5)
                    .map(|(i, _n)| i as u32)
                    .take(1),
                &mut flags,
            );

            builder.partial_rebuild(
                &mut bvh,
                &mut temp_bvh,
                |node_id| flags[node_id],
                PlocSearchDistance::Minimum,
                SortPrecision::U64,
                0,
            );

            bvh.validate(&tris, false, true);
        }
    }
}
