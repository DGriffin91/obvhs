use std::borrow::Borrow;

use crate::{
    bvh2::Bvh2,
    fast_stack,
    faststack::FastStack,
    ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
};

/// Provide a set of leaves that need to be rebuilt. After calling, flags will be true for any node index up the tree
/// that needs to be included in the partial rebuild.
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
    /// Fully rebuild the bvh from its current leaves.
    ///
    /// # Arguments
    /// * `bvh` - An existing bvh with valid leaves. Inner nodes are ignored.
    /// * `search_distance` - Which search distance should be used when building the ploc.
    /// * `sort_precision` - Bits used for ploc radix sort. More bits results in a more accurate but slower sort.
    /// * `search_depth_threshold` - Below this depth a search distance of 1 will be used. Set to 0 to bypass and
    ///   just use search_distance.
    pub fn full_rebuild(
        &mut self,
        bvh: &mut Bvh2,
        search_distance: PlocSearchDistance,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        if bvh.nodes.len() < 2 {
            return;
        }

        self.current_nodes.clear();

        // Collect all leaves
        for node in &bvh.nodes {
            if node.is_leaf() {
                self.current_nodes.push(*node);
            }
        }

        self.rebuild_from_leaves::<false>(
            bvh,
            search_distance,
            sort_precision,
            search_depth_threshold,
        );
    }

    /// Partially rebuild the bvh. The given set of leaves and the subtrees that do not include any of the given leaves
    /// will be built into a new bvh. If the set of leaves is a small enough proportion of the total this can be faster
    /// since there may be large portions of the BVH that don't need to be updated. If the proportion is very high it
    /// can be faster to build from scratch instead, avoiding the overhead of doing a partial rebuild. If only a few
    /// nodes need to be updated it might be faster and produce a better BVH to selectively reinsert them.
    ///
    /// # Arguments
    /// * `bvh` - An existing bvh with valid layout (AABBs in the tree above nodes that are to be rebuilt does not need
    ///   to be correct)
    /// * `should_remove()` - should return true for any node that should be include in the rebuild. This includes the
    ///   entire chain up from any leaves that need to be updated. Use PlocBuilder::compute_rebuild_path_flags() or
    ///   similar to compute. The leaves should have their new AABB before calling partial_rebuild() but the BVH does
    ///   not need to be refit to accommodate them.
    /// * `search_distance` - Which search distance should be used when building the ploc.
    /// * `sort_precision` - Bits used for ploc radix sort. More bits results in a more accurate but slower sort.
    /// * `search_depth_threshold` - Below this depth a search distance of 1 will be used. Set to 0 to bypass and
    ///   just use search_distance.
    pub fn partial_rebuild(
        &mut self,
        bvh: &mut Bvh2,
        should_remove: impl Fn(usize) -> bool,
        search_distance: PlocSearchDistance,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        if bvh.nodes.len() < 2 {
            return;
        }

        self.current_nodes.clear();

        // Top down traverse to collect leaves and unflagged subtrees
        fast_stack!(u32, (96, 192), bvh.max_depth, stack, {
            stack.push(bvh.nodes[0].first_index);
            while let Some(left_node_index) = stack.pop() {
                for node_index in [left_node_index as usize, left_node_index as usize + 1] {
                    let node = &mut bvh.nodes[node_index];
                    if !should_remove(node_index) || node.is_leaf() {
                        self.current_nodes.push(*node);
                    } else {
                        stack.push(node.first_index);
                    }
                    node.set_invalid();
                }
            }
        });

        self.rebuild_from_leaves::<true>(
            bvh,
            search_distance,
            sort_precision,
            search_depth_threshold,
        );
    }

    fn rebuild_from_leaves<const PARTIAL: bool>(
        &mut self,
        bvh: &mut Bvh2,
        search_distance: PlocSearchDistance,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        if bvh.nodes.len() < 2 {
            return;
        }

        let had_parents = !bvh.parents.is_empty();
        let had_primitives_to_nodes = !bvh.primitives_to_nodes.is_empty();

        self.next_nodes.clear();
        self.mortons.clear();

        // Rebuild BVH from leaves
        let total_aabb = *bvh.nodes[0].aabb();
        let sdt = search_depth_threshold;
        match search_distance {
            PlocSearchDistance::Minimum => {
                self.build_ploc_from_leaves::<1, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::VeryLow => {
                self.build_ploc_from_leaves::<2, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::Low => {
                self.build_ploc_from_leaves::<6, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::Medium => {
                self.build_ploc_from_leaves::<14, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::High => {
                self.build_ploc_from_leaves::<24, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
            PlocSearchDistance::VeryHigh => {
                self.build_ploc_from_leaves::<32, PARTIAL>(bvh, total_aabb, sort_precision, sdt)
            }
        }

        if had_parents {
            bvh.update_parents();
        }
        if had_primitives_to_nodes {
            bvh.update_primitives_to_nodes();
        }
    }
}

#[cfg(test)]
mod tests {

    use glam::UVec2;

    use super::*;
    use crate::{
        INVALID,
        test_util::{geometry::demoscene, sampling::hash_noise},
    };

    #[test]
    fn test_full_rebuild() {
        let sm = demoscene(5, 0);
        for tris in [&demoscene(31, 0), &sm, &sm[..1], &sm[..2], &sm[..3], &[]] {
            let mut builder = PlocBuilder::with_capacity(tris.len());

            let mut bvh = builder.build(
                PlocSearchDistance::Minimum,
                tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(tris, false, true);

            builder.full_rebuild(&mut bvh, PlocSearchDistance::Minimum, SortPrecision::U64, 1);

            bvh.validate(tris, false, true);
        }
    }

    #[test]
    fn test_full_rebuild_with_free_indices() {
        let tris = demoscene(32, 0);

        let mut builder = PlocBuilder::with_capacity(tris.len());

        let mut bvh = builder.build(
            PlocSearchDistance::Minimum,
            &tris,
            (0..tris.len() as u32).collect::<Vec<_>>(),
            SortPrecision::U64,
            1,
        );

        bvh.validate(&tris, false, true);

        // Remove some primitives to create free indices
        bvh.remove_primitive(6);
        bvh.remove_primitive(9);
        bvh.remove_primitive(10);
        assert_eq!(bvh.primitive_indices_freelist.len(), 3);
        assert!(bvh.primitive_indices.contains(&INVALID));

        // Now do a full rebuild
        builder.full_rebuild(&mut bvh, PlocSearchDistance::Minimum, SortPrecision::U64, 1);

        bvh.validate(&tris, false, true);
    }

    #[test]
    fn test_partial_rebuild_with_all_leaves() {
        let sm = demoscene(5, 0);
        for tris in [&demoscene(31, 0), &sm, &sm[..1], &sm[..2], &sm[..3], &[]] {
            let mut builder = PlocBuilder::with_capacity(tris.len());

            let mut bvh = builder.build(
                PlocSearchDistance::Minimum,
                tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(tris, false, true);

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
                |node_id| flags[node_id],
                PlocSearchDistance::Minimum,
                SortPrecision::U64,
                1,
            );

            bvh.validate(tris, false, true);
        }
    }

    #[test]
    fn test_partial_rebuild_with_one_leaf() {
        let tris = demoscene(8, 0);

        let mut builder = PlocBuilder::with_capacity(tris.len());

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

            let mut bvh = builder.build(
                PlocSearchDistance::Minimum,
                tris,
                (0..tris.len() as u32).collect::<Vec<_>>(),
                SortPrecision::U64,
                1,
            );

            bvh.validate(tris, false, true);

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
                |node_id| flags[node_id],
                PlocSearchDistance::Minimum,
                SortPrecision::U64,
                0,
            );

            bvh.validate(tris, false, true);
        }
    }
}
