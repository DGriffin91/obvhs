//! PLOC (Parallel, Locally Ordered Clustering) BVH 2 Builder.

pub mod morton;

// https://madmann91.github.io/2021/05/05/ploc-revisited.html
// https://github.com/meistdan/ploc/
// https://meistdan.github.io/publications/ploc/paper.pdf
// https://github.com/madmann91/bvh/blob/v1/include/bvh/locally_ordered_clustering_builder.hpp

use glam::DVec3;
use rdst::{RadixKey, RadixSort};

#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::bvh2::node::Bvh2Node;
use crate::ploc::morton::{morton_encode_u128_unorm, morton_encode_u64_unorm};
use crate::{aabb::Aabb, bvh2::Bvh2};

impl PlocSearchDistance {
    /// # Arguments
    /// * `aabbs` - A list of bounding boxes. Should correspond to the number and order of primitives.
    /// * `indices` - The list indices used to index into the list of primitives. This allows for
    ///   flexibility in which primitives are included in the bvh and in what order they are referenced.
    ///   Often this would just be equivalent to: (0..aabbs.len() as u32).collect::<Vec<_>>()
    /// * `sort_precision` - Bits used for ploc radix sort. More bits results in a more accurate but slower sort.
    /// * `search_depth_threshold` - Below this depth a search distance of 1 will be used. Set to 0 to bypass and
    ///   just use PlocSearchDistance. When trying to optimize build time it can be beneficial to limit the search
    ///   distance for the first few passes as that is when the largest number of primitives are being considered.
    ///   This pairs are initially found more quickly since it doesn't need to search as far, and they are also
    ///   found more often, since the nodes need to both agree to become paired. This also seems to occasionally
    ///   result in an overall better bvh structure.
    pub fn build(
        &self,
        aabbs: &[Aabb],
        indices: Vec<u32>,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) -> Bvh2 {
        match self {
            PlocSearchDistance::Minimum => {
                build_ploc::<1>(aabbs, indices, sort_precision, search_depth_threshold)
            }
            PlocSearchDistance::VeryLow => {
                build_ploc::<2>(aabbs, indices, sort_precision, search_depth_threshold)
            }
            PlocSearchDistance::Low => {
                build_ploc::<6>(aabbs, indices, sort_precision, search_depth_threshold)
            }
            PlocSearchDistance::Medium => {
                build_ploc::<14>(aabbs, indices, sort_precision, search_depth_threshold)
            }
            PlocSearchDistance::High => {
                build_ploc::<24>(aabbs, indices, sort_precision, search_depth_threshold)
            }
            PlocSearchDistance::VeryHigh => {
                build_ploc::<32>(aabbs, indices, sort_precision, search_depth_threshold)
            }
        }
    }
}

/// # Arguments
/// * `aabbs` - A list of bounding boxes. Should correspond to the number and order of primitives.
/// * `search_depth_threshold` - Below this depth a search distance of 1 will be used
/// * `sort_precision` - Bits used for ploc radix sort. More bits results in a more accurate but slower sort.
/// * `search_depth_threshold` - Below this depth a search distance of 1 will be used. Set to 0 to bypass and
///   just use SEARCH_DISTANCE.
///
/// SEARCH_DISTANCE should be <= 32
pub fn build_ploc<const SEARCH_DISTANCE: usize>(
    aabbs: &[Aabb],
    indices: Vec<u32>,
    sort_precision: SortPrecision,
    search_depth_threshold: usize,
) -> Bvh2 {
    crate::scope!("build_ploc");

    let prim_count = aabbs.len();

    if prim_count == 0 {
        return Bvh2::default();
    }

    let mut total_aabb = Aabb::empty();

    // TODO perf, could make parallel if each thread tracks its own min/max and then combines afterward.
    // (Or use atomics but idk if contention would be an issue)
    let init_leafs = indices
        .iter()
        .enumerate()
        .map(|(i, prim_index)| {
            let aabb = aabbs[i];
            debug_assert!(!aabb.min.is_nan());
            debug_assert!(!aabb.max.is_nan());
            total_aabb.extend(aabb.min);
            total_aabb.extend(aabb.max);
            Bvh2Node {
                aabb,
                prim_count: 1,
                first_index: *prim_index,
            }
        })
        .collect::<Vec<_>>();

    let nodes = build_ploc_from_leafs::<SEARCH_DISTANCE>(
        init_leafs,
        total_aabb,
        sort_precision,
        search_depth_threshold,
    );

    Bvh2 {
        nodes,
        primitive_indices: indices,
        children_are_ordered_after_parents: true,
        ..Default::default()
    }
}

pub fn build_ploc_from_leafs<const SEARCH_DISTANCE: usize>(
    mut current_nodes: Vec<Bvh2Node>,
    total_aabb: Aabb,
    sort_precision: SortPrecision,
    search_depth_threshold: usize,
) -> Vec<Bvh2Node> {
    crate::scope!("build_ploc_from_leafs");

    let prim_count = current_nodes.len();

    // Merge nodes until there is only one left
    let nodes_count = (2 * prim_count as i64 - 1).max(0) as usize;

    let scale = 1.0 / total_aabb.diagonal().as_dvec3();
    let offset = -total_aabb.min.as_dvec3() * scale;

    // Sort primitives according to their morton code
    sort_precision.sort_nodes(&mut current_nodes, scale, offset);

    let mut nodes = vec![Bvh2Node::default(); nodes_count];

    let mut insert_index = nodes_count;
    let mut next_nodes = Vec::with_capacity(prim_count);
    assert!(i8::MAX as usize > SEARCH_DISTANCE);
    let mut merge: Vec<i8> = vec![0; prim_count];

    #[cfg(not(feature = "parallel"))]
    let mut cache = SearchCache::<SEARCH_DISTANCE>::default();

    let mut depth: usize = 0;
    while current_nodes.len() > 1 {
        if SEARCH_DISTANCE == 1 || depth < search_depth_threshold {
            // TODO try making build_ploc_from_leafs_one that embeds this logic into
            // the main `while index < merge.len() {` loop  (may not be faster, tbd)
            let mut last_cost = f32::MAX;
            let count = current_nodes.len() - 1;
            assert!(count < merge.len()); // Try to elide bounds check
            (0..count).for_each(|i| {
                let cost = current_nodes[i]
                    .aabb
                    .union(&current_nodes[i + 1].aabb)
                    .half_area();
                merge[i] = if last_cost < cost { -1 } else { 1 };
                last_cost = cost;
            });
            merge[current_nodes.len() - 1] = -1;
        } else {
            #[cfg(feature = "parallel")]
            let iter = merge.par_iter_mut();
            #[cfg(not(feature = "parallel"))]
            let iter = merge.iter_mut();
            iter.enumerate()
                .take(current_nodes.len())
                .for_each(|(index, best)| {
                    #[cfg(feature = "parallel")]
                    {
                        *best = find_best_node_basic(index, &current_nodes, SEARCH_DISTANCE);
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        *best = cache.find_best_node(index, &current_nodes);
                    }
                });
        };

        let mut index = 0;
        while index < current_nodes.len() {
            let index_offset = merge[index] as i64;
            let best_index = (index as i64 + index_offset) as usize;
            // The two nodes should be merged if they agree on their respective merge indices.
            if best_index as i64 + merge[best_index] as i64 != index as i64 {
                // If not, the current node should be kept for the next iteration
                next_nodes.push(current_nodes[index]);
                index += 1;
                continue;
            }

            // Since we only need to merge once, we only merge if the first index is less than the second.
            if best_index > index {
                index += 1;
                continue;
            }

            debug_assert_ne!(best_index, index);

            let left = current_nodes[index];
            let right = current_nodes[best_index];

            // Reserve space in the target array for the two children
            debug_assert!(insert_index >= 2);
            insert_index -= 2;

            // Create the parent node and place it in the array for the next iteration
            next_nodes.push(Bvh2Node {
                aabb: left.aabb.union(&right.aabb),
                prim_count: 0,
                first_index: insert_index as u32,
            });

            // Out of bounds here error here could indicate NaN present in input aabb. Try running in debug mode.
            nodes[insert_index] = left;
            nodes[insert_index + 1] = right;

            if SEARCH_DISTANCE == 1 && index_offset == 1 {
                // If the search distance is only 1, and the next index was merged with this one,
                // we can skip the next index.
                // The code for this with the while loop seemed to also be slightly faster than:
                //     for (index, best_index) in merge.iter().enumerate() {
                // even in the other cases. For some reason...
                index += 2;
            } else {
                index += 1;
            }
        }

        (next_nodes, current_nodes) = (current_nodes, next_nodes);
        next_nodes.clear();
        depth += 1;
    }

    insert_index = insert_index.saturating_sub(1);
    nodes[insert_index] = current_nodes[0];
    nodes
}

#[cfg(feature = "parallel")]
fn find_best_node_basic(index: usize, nodes: &[Bvh2Node], search_distance: usize) -> i8 {
    let mut best_node = index;
    let mut best_cost = f32::INFINITY;

    let begin = index - search_distance.min(index);
    let end = (index + search_distance + 1).min(nodes.len());

    let our_aabb = nodes[index].aabb;
    for other in begin..end {
        if other == index {
            continue;
        }
        let cost = our_aabb.union(&nodes[other].aabb).half_area();
        if cost < best_cost {
            best_node = other;
            best_cost = cost;
        }
    }

    (best_node as i64 - index as i64) as i8
}

/// In PLOC, the number of nodes before and after the current one that are evaluated for pairing.
/// Minimum (1) has a fast path in building and still results in decent quality BVHs especially
/// when paired with a bit of reinsertion.
#[derive(Default, Clone, Copy)]
pub enum PlocSearchDistance {
    /// 1
    Minimum,
    /// 2
    VeryLow,
    /// 6
    Low,
    #[default]
    /// 14
    Medium,
    /// 24
    High,
    /// 32
    VeryHigh,
}

impl From<u32> for PlocSearchDistance {
    fn from(value: u32) -> Self {
        match value {
            1 => PlocSearchDistance::Minimum,
            2 => PlocSearchDistance::VeryLow,
            6 => PlocSearchDistance::Low,
            14 => PlocSearchDistance::Medium,
            24 => PlocSearchDistance::High,
            32 => PlocSearchDistance::VeryHigh,
            _ => panic!("Invalid value for PlocSearchDistance: {value}"),
        }
    }
}

// Tried using a Vec it was ~30% slower with a search distance of 14.
// Tried making the Vec flat, used get_unchecked, etc... (without those it was ~80% slower)
pub struct SearchCache<const SEARCH_DISTANCE: usize>([[f32; SEARCH_DISTANCE]; SEARCH_DISTANCE]);

impl<const SEARCH_DISTANCE: usize> Default for SearchCache<SEARCH_DISTANCE> {
    fn default() -> Self {
        SearchCache([[0.0; SEARCH_DISTANCE]; SEARCH_DISTANCE])
    }
}

impl<const SEARCH_DISTANCE: usize> SearchCache<SEARCH_DISTANCE> {
    #[inline]
    #[cfg(not(feature = "parallel"))]
    fn back(&self, index: usize, other: usize) -> f32 {
        // Note: the compiler removes the bounds check due to the % SEARCH_DISTANCE
        self.0[other % SEARCH_DISTANCE][index % SEARCH_DISTANCE]
    }

    #[inline]
    #[cfg(not(feature = "parallel"))]
    fn front(&mut self, index: usize, other: usize) -> &mut f32 {
        &mut self.0[index % SEARCH_DISTANCE][other % SEARCH_DISTANCE]
    }

    #[cfg(not(feature = "parallel"))]
    fn find_best_node(&mut self, index: usize, nodes: &[Bvh2Node]) -> i8 {
        let mut best_node = index;
        let mut best_cost = f32::INFINITY;

        let begin = index - SEARCH_DISTANCE.min(index);
        let end = (index + SEARCH_DISTANCE + 1).min(nodes.len());

        for other in begin..index {
            let area = self.back(index, other);
            if area < best_cost {
                best_node = other;
                best_cost = area;
            }
        }

        let our_aabb = nodes[index].aabb;
        ((index + 1)..end).for_each(|other| {
            let cost = our_aabb.union(&nodes[other].aabb).half_area();
            *self.front(index, other) = cost;
            if cost < best_cost {
                best_node = other;
                best_cost = cost;
            }
        });

        (best_node as i64 - index as i64) as i8
    }
}

// ---------------------
// --- SORTING NODES ---
// ---------------------

// TODO find a not terrible way to make this less repetitive

#[derive(Debug, Copy, Clone)]
pub enum SortPrecision {
    U128,
    U64,
}

impl SortPrecision {
    fn sort_nodes(&self, current_nodes: &mut Vec<Bvh2Node>, scale: DVec3, offset: DVec3) {
        match self {
            SortPrecision::U128 => sort_nodes_m128(current_nodes, scale, offset),
            SortPrecision::U64 => sort_nodes_m64(current_nodes, scale, offset),
        }
    }
}

#[derive(Clone, Copy)]
struct Morton128 {
    index: usize,
    code: u128,
}

impl RadixKey for Morton128 {
    const LEVELS: usize = 16;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        self.code.get_level(level)
    }
}

#[derive(Clone, Copy)]
struct Morton64 {
    index: usize,
    code: u64,
}

impl RadixKey for Morton64 {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        self.code.get_level(level)
    }
}

fn sort_nodes_m128(current_nodes: &mut Vec<Bvh2Node>, scale: DVec3, offset: DVec3) {
    crate::scope!("sort_nodes_m128");

    let morton_code_proc = |(index, leaf): (usize, &Bvh2Node)| {
        let center = leaf.aabb.center().as_dvec3() * scale + offset;
        Morton128 {
            index,
            code: morton_encode_u128_unorm(center),
        }
    };

    // TODO perf/forte Due to rayon overhead using par_iter can be slower than just iter for small quantities of nodes.
    // 100k chosen from testing various tri counts with the demoscene example
    #[cfg(feature = "parallel")]
    let max_parallel = 100_000;

    let indexed_mortons = &mut Vec::with_capacity(current_nodes.len());

    #[cfg(feature = "parallel")]
    {
        if current_nodes.len() > max_parallel {
            *indexed_mortons = current_nodes
                .par_iter()
                .enumerate()
                .map(morton_code_proc)
                .collect();
        } else {
            *indexed_mortons = current_nodes
                .iter()
                .enumerate()
                .map(morton_code_proc)
                .collect();
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        *indexed_mortons = current_nodes
            .iter()
            .enumerate()
            .map(morton_code_proc)
            .collect();
    }

    indexed_mortons.radix_sort_unstable();

    #[cfg(feature = "parallel")]
    {
        if current_nodes.len() > max_parallel {
            *current_nodes = indexed_mortons
                .into_par_iter()
                .map(|m| current_nodes[m.index])
                .collect();
        } else {
            *current_nodes = indexed_mortons
                .iter_mut()
                .map(|m| current_nodes[m.index])
                .collect();
        };
    }

    #[cfg(not(feature = "parallel"))]
    {
        *current_nodes = indexed_mortons
            .into_iter()
            .map(|m| current_nodes[m.index])
            .collect();
    }
}

fn sort_nodes_m64(current_nodes: &mut Vec<Bvh2Node>, scale: DVec3, offset: DVec3) {
    crate::scope!("sort_nodes_m64");

    let morton_code_proc = |(index, leaf): (usize, &Bvh2Node)| {
        let center = leaf.aabb.center().as_dvec3() * scale + offset;
        Morton64 {
            index,
            code: morton_encode_u64_unorm(center),
        }
    };

    // TODO perf/forte Due to rayon overhead using par_iter can be slower than just iter for small quantities of nodes.
    // 100k chosen from testing various tri counts with the demoscene example
    #[cfg(feature = "parallel")]
    let max_parallel = 100_000;

    let indexed_mortons = &mut Vec::with_capacity(current_nodes.len());

    #[cfg(feature = "parallel")]
    {
        if current_nodes.len() > max_parallel {
            *indexed_mortons = current_nodes
                .par_iter()
                .enumerate()
                .map(morton_code_proc)
                .collect();
        } else {
            *indexed_mortons = current_nodes
                .iter()
                .enumerate()
                .map(morton_code_proc)
                .collect();
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        *indexed_mortons = current_nodes
            .iter()
            .enumerate()
            .map(morton_code_proc)
            .collect();
    }

    indexed_mortons.radix_sort_unstable();

    #[cfg(feature = "parallel")]
    {
        if current_nodes.len() > max_parallel {
            *current_nodes = indexed_mortons
                .into_par_iter()
                .map(|m| current_nodes[m.index])
                .collect();
        } else {
            *current_nodes = indexed_mortons
                .iter_mut()
                .map(|m| current_nodes[m.index])
                .collect();
        };
    }

    #[cfg(not(feature = "parallel"))]
    {
        *current_nodes = indexed_mortons
            .into_iter()
            .map(|m| current_nodes[m.index])
            .collect();
    }
}
