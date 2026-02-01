//! PLOC (Parallel, Locally Ordered Clustering) BVH 2 Builder.

pub mod morton;
pub mod rebuild;

// https://madmann91.github.io/2021/05/05/ploc-revisited.html
// https://github.com/meistdan/ploc/
// https://meistdan.github.io/publications/ploc/paper.pdf
// https://github.com/madmann91/bvh/blob/v1/include/bvh/locally_ordered_clustering_builder.hpp

use std::{f32, mem};

use bytemuck::{Pod, Zeroable, cast_slice_mut, zeroed_vec};
use glam::DVec3;
use rdst::RadixKey;

#[cfg(feature = "parallel")]
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

#[cfg(not(feature = "parallel"))]
use rdst::RadixSort;

use crate::bvh2::DEFAULT_MAX_STACK_DEPTH;
use crate::ploc::morton::{morton_encode_u64_unorm, morton_encode_u128_unorm};
use crate::{Boundable, bvh2::node::Bvh2Node};
use crate::{aabb::Aabb, bvh2::Bvh2};

#[derive(Clone)]
pub struct PlocBuilder {
    pub current_nodes: Vec<Bvh2Node>,
    pub next_nodes: Vec<Bvh2Node>,

    // Enough space/align for Morton64 or Morton128. If this is updated make sure to also update anything that uses it.
    // As things depend on it being exactly Vec<[u128; 2]>
    pub mortons: Vec<[u128; 2]>,

    #[cfg(feature = "parallel")]
    pub local_aabbs: Vec<Aabb>,
}

impl Default for PlocBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PlocBuilder {
    /// Initialize a ploc builder. After initial building, keep around this builder to reuse the associated allocations.
    pub fn new() -> PlocBuilder {
        crate::scope!("preallocate_builder");
        PlocBuilder {
            current_nodes: Vec::new(),
            next_nodes: Vec::new(),
            mortons: Vec::new(),

            #[cfg(feature = "parallel")]
            local_aabbs: Vec::new(),
        }
    }

    /// Initialize a ploc builder with pre-allocated capacity for building a bvh with prim_count.
    /// After initial building, keep around this builder to reuse the associated allocations.
    pub fn with_capacity(prim_count: usize) -> PlocBuilder {
        crate::scope!("preallocate_builder");
        PlocBuilder {
            current_nodes: zeroed_vec(prim_count),
            next_nodes: zeroed_vec(prim_count),
            mortons: zeroed_vec(prim_count),

            #[cfg(feature = "parallel")]
            local_aabbs: zeroed_vec(128),
        }
    }

    /// # Arguments
    /// * `search_distance` - Which search distance should be used when building the ploc.
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
    #[inline]
    pub fn build<T: Boundable>(
        &mut self,
        search_distance: PlocSearchDistance,
        aabbs: &[T],
        indices: Vec<u32>,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) -> Bvh2 {
        let mut bvh = Bvh2::zeroed(aabbs.len());
        self.build_with_bvh(
            &mut bvh,
            search_distance,
            aabbs,
            indices,
            sort_precision,
            search_depth_threshold,
        );
        bvh
    }

    /// # Arguments
    /// * `bvh` - An existing bvh. The builder will clear this bvh and reuse its allocations.
    /// * `search_distance` - Which search distance should be used when building the ploc.
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
    pub fn build_with_bvh<T: Boundable>(
        &mut self,
        bvh: &mut Bvh2,
        search_distance: PlocSearchDistance,
        aabbs: &[T],
        indices: Vec<u32>,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        let search_thresh = search_depth_threshold;
        match search_distance {
            PlocSearchDistance::Minimum => {
                self.build_ploc::<1, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
            PlocSearchDistance::VeryLow => {
                self.build_ploc::<2, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
            PlocSearchDistance::Low => {
                self.build_ploc::<6, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
            PlocSearchDistance::Medium => {
                self.build_ploc::<14, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
            PlocSearchDistance::High => {
                self.build_ploc::<24, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
            PlocSearchDistance::VeryHigh => {
                self.build_ploc::<32, T>(bvh, aabbs, indices, sort_precision, search_thresh)
            }
        }
    }

    /// # Arguments
    /// * `bvh` - An existing bvh. The builder will clear this bvh and reuse its allocations.
    /// * `aabbs` - A list of bounding boxes. Should correspond to the number and order of primitives.
    /// * `sort_precision` - Bits used for ploc radix sort. More bits results in a more accurate but slower sort.
    /// * `search_depth_threshold` - Below this depth a search distance of 1 will be used. Set to 0 to bypass and
    ///   just use SEARCH_DISTANCE.
    ///
    /// SEARCH_DISTANCE should be <= 32
    pub fn build_ploc<const SEARCH_DISTANCE: usize, T: Boundable>(
        &mut self,
        bvh: &mut Bvh2,
        aabbs: &[T],
        indices: Vec<u32>,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        crate::scope!("build_ploc");

        let prim_count = aabbs.len();

        bvh.reset_for_reuse(prim_count, Some(indices));

        if prim_count == 0 {
            return;
        }

        #[inline]
        fn init_node(prim_index: &u32, aabb: Aabb, local_aabb: &mut Aabb) -> Bvh2Node {
            local_aabb.extend(aabb.min);
            local_aabb.extend(aabb.max);
            debug_assert!(!aabb.min.is_nan());
            debug_assert!(!aabb.max.is_nan());
            Bvh2Node::new(aabb, 1, *prim_index)
        }

        let mut total_aabb = None;
        self.current_nodes.resize(prim_count, Default::default());

        // TODO perf/forte Due to rayon overhead using par_iter can be slower than just iter for small quantities of nodes.
        // 500k chosen from testing various tri counts with the demoscene example
        #[cfg(feature = "parallel")]
        let min_parallel = 500_000;

        #[cfg(feature = "parallel")]
        if prim_count >= min_parallel {
            let chunk_size = aabbs.len().div_ceil(rayon::current_num_threads());

            let chunks = self
                .current_nodes
                .par_iter_mut()
                .zip(&bvh.primitive_indices)
                .enumerate()
                .chunks(chunk_size);

            self.local_aabbs.resize(chunks.len(), Aabb::empty());

            chunks
                .zip(self.local_aabbs.par_iter_mut())
                .for_each(|(data, local_aabb)| {
                    for (i, (node, prim_index)) in data {
                        *node = init_node(prim_index, aabbs[i].aabb(), local_aabb);
                    }
                });

            let mut total = Aabb::empty();
            for local_aabb in self.local_aabbs.iter_mut() {
                total.extend(local_aabb.min);
                total.extend(local_aabb.max);
            }
            total_aabb = Some(total);
        }

        if total_aabb.is_none() {
            let mut total = Aabb::empty();
            self.current_nodes
                .iter_mut()
                .zip(&bvh.primitive_indices)
                .zip(aabbs)
                .for_each(|((node, prim_index), aabb)| {
                    *node = init_node(prim_index, aabb.aabb(), &mut total)
                });
            total_aabb = Some(total);
        }

        self.build_ploc_from_leaves::<SEARCH_DISTANCE, false>(
            bvh,
            total_aabb.unwrap(),
            sort_precision,
            search_depth_threshold,
        );
    }

    /// Prefer using Bvh2::build(), Bvh2::build_with_bvh(), Bvh2::build_ploc(), Bvh2::partial_rebuild(),
    /// or Bvh2::full_rebuild(). This is only public for non-typical usages.
    /// REBUILD is for partial BVH rebuilds. In that case inner nodes should be freed by setting them to invalid
    /// (with Bvh2Node::set_invalid()) and both respective inner and leaf nodes moved on to PlocBuilder::current_nodes.
    /// They must always be removed in pairs with the starting on an odd number. See PlocBuilder::partial_rebuild()
    pub fn build_ploc_from_leaves<const SEARCH_DISTANCE: usize, const REBUILD: bool>(
        &mut self,
        bvh: &mut Bvh2,
        total_aabb: Aabb,
        sort_precision: SortPrecision,
        search_depth_threshold: usize,
    ) {
        crate::scope!("build_ploc_from_leaves");

        let prim_count = self.current_nodes.len();

        if prim_count == 0 {
            return;
        }

        // Merge nodes until there is only one left
        let nodes_count = (2 * prim_count as i64 - 1).max(0) as usize;

        let mut insert_index = if REBUILD {
            if bvh.nodes.is_empty() {
                return;
            }
            assert!(bvh.nodes.len() >= nodes_count);
            bvh.nodes.len() - 1
        } else {
            bvh.nodes.resize(nodes_count, Bvh2Node::default());
            nodes_count
        };

        let scale = 1.0 / total_aabb.diagonal().as_dvec3();
        let offset = -total_aabb.min.as_dvec3() * scale;

        let mortons_size = match sort_precision {
            SortPrecision::U128 => prim_count,
            SortPrecision::U64 => prim_count.div_ceil(2),
        };
        self.mortons.resize(mortons_size, Default::default());
        self.next_nodes.resize(prim_count, Default::default());

        // Sort primitives according to their morton code
        sort_precision.sort_nodes(
            &mut self.current_nodes,
            &mut self.next_nodes,
            &mut self.mortons,
            scale,
            offset,
        );
        mem::swap(&mut self.current_nodes, &mut self.next_nodes);

        assert!(i8::MAX as usize > SEARCH_DISTANCE);

        let merge_buffer: &mut [i8] = &mut cast_slice_mut(&mut self.mortons)[..prim_count];

        #[cfg(not(feature = "parallel"))]
        let mut cache = SearchCache::<SEARCH_DISTANCE>::default();

        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads();

        #[cfg(feature = "parallel")]
        let mut cache = if prim_count < 4000 {
            vec![SearchCache::<SEARCH_DISTANCE>::default()]
        } else {
            vec![SearchCache::<SEARCH_DISTANCE>::default(); threads * 4]
        };

        let mut depth: usize = 0;
        let mut next_nodes_idx = 0;
        let mut count = prim_count;
        while count > 1 {
            let merge = &mut merge_buffer[..count];
            if SEARCH_DISTANCE == 1 || depth < search_depth_threshold {
                let mut last_cost = f32::INFINITY;
                let calculate_costs = |(i, merge_n): (usize, &mut i8)| {
                    let cost = self.current_nodes[i]
                        .aabb()
                        .union(self.current_nodes[i + 1].aabb())
                        .half_area();
                    *merge_n = if last_cost < cost { -1 } else { 1 };
                    last_cost = cost;
                };

                let count_m1 = count - 1;
                let merge_m1 = &mut merge[..count_m1];

                #[cfg(feature = "parallel")]
                {
                    let chunk_size = merge_m1.len().div_ceil(threads);
                    let calculate_costs_parallel = |(chunk_id, chunk): (usize, &mut [i8])| {
                        let start = chunk_id * chunk_size;
                        let mut last_cost = if start == 0 {
                            f32::INFINITY
                        } else {
                            self.current_nodes[start - 1]
                                .aabb()
                                .union(self.current_nodes[start].aabb())
                                .half_area()
                        };
                        for (local_n, merge_n) in chunk.iter_mut().enumerate() {
                            let i = local_n + start;
                            let cost = self.current_nodes[i]
                                .aabb()
                                .union(self.current_nodes[i + 1].aabb())
                                .half_area();
                            *merge_n = if last_cost < cost { -1 } else { 1 };
                            last_cost = cost;
                        }
                    };

                    // TODO perf/forte Due to rayon overhead using par_iter can be slower than just iter for small quantities.
                    // 300k chosen from testing various scenes in tray racing
                    if count < 300_000 {
                        merge_m1.iter_mut().enumerate().for_each(calculate_costs);
                    } else {
                        merge_m1
                            .par_chunks_mut(chunk_size.max(1))
                            .enumerate()
                            .for_each(calculate_costs_parallel)
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    merge_m1.iter_mut().enumerate().for_each(calculate_costs);
                }
                merge[count_m1] = -1;
            } else {
                #[cfg(not(feature = "parallel"))]
                merge.iter_mut().enumerate().for_each(|(index, best)| {
                    *best = cache.find_best_node(index, &self.current_nodes[..count]);
                });

                #[cfg(feature = "parallel")]
                {
                    // TODO perf/forte Due to rayon overhead using par_iter can be slower than just iter for small quantities.
                    // 4k chosen from testing with demoscene
                    if count < 4000 {
                        let cache = &mut cache[0];
                        merge.iter_mut().enumerate().for_each(|(index, best)| {
                            *best = cache.find_best_node(index, &self.current_nodes[..count]);
                        });
                    } else {
                        // Split search into chunks in parallel
                        let chunk_size = merge.len().div_ceil(cache.len());
                        let chunks = merge.par_chunks_mut(merge.len().div_ceil(cache.len()));
                        if chunks.len() > cache.len() {
                            cache.resize(chunks.len(), SearchCache::<SEARCH_DISTANCE>::default());
                        }
                        chunks.zip(cache.par_iter_mut()).enumerate().for_each(
                            |(chunk, (bests, cache))| {
                                for (i, best) in bests.iter_mut().enumerate() {
                                    let index = chunk * chunk_size + i;
                                    *best = cache.find_best_node_parallel(
                                        index,
                                        i,
                                        &self.current_nodes[..count],
                                    );
                                }
                            },
                        );
                    }
                }
            };
            let mut index = 0;
            // Tried making this parallel but it was similar perf as the sequential version below. Could be memory bound?
            // https://github.com/DGriffin91/pool_racing/commit/a35b92496a1c28043b11565ee48dff0137ada68f
            while index < count {
                let index_offset = merge[index] as i64;
                let best_index = (index as i64 + index_offset) as usize;
                // The two nodes should be merged if they agree on their respective merge indices.
                if best_index as i64 + merge[best_index] as i64 != index as i64 {
                    // If not, the current node should be kept for the next iteration
                    self.next_nodes[next_nodes_idx] = self.current_nodes[index];
                    next_nodes_idx += 1;
                    index += 1;
                    continue;
                }

                // Since we only need to merge once, we only merge if the first index is less than the second.
                if best_index > index {
                    index += 1;
                    continue;
                }

                debug_assert_ne!(best_index, index);

                let left = self.current_nodes[index];
                let right = self.current_nodes[best_index];

                let first_child;

                // Reserve space in the target array for the two children
                if REBUILD {
                    loop {
                        // Out of bounds here error here could indicate NaN present in input aabb. Try running in debug mode.
                        let left_slot = &mut bvh.nodes[insert_index - 1];
                        if !left_slot.valid() {
                            *left_slot = left;
                            debug_assert!(!bvh.nodes[insert_index].valid());
                            bvh.nodes[insert_index] = right;
                            first_child = insert_index - 1;
                            insert_index -= 2;
                            break;
                        }
                        insert_index -= 2;
                    }
                } else {
                    debug_assert!(insert_index >= 2);
                    insert_index -= 2;
                    // Out of bounds here error here could indicate NaN present in input aabb. Try running in debug mode.
                    bvh.nodes[insert_index] = left;
                    bvh.nodes[insert_index + 1] = right;
                    first_child = insert_index;
                }

                // Create the parent node and place it in the array for the next iteration
                self.next_nodes[next_nodes_idx] =
                    Bvh2Node::new(left.aabb().union(right.aabb()), 0, first_child as u32);
                next_nodes_idx += 1;

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

            mem::swap(&mut self.next_nodes, &mut self.current_nodes);
            count = next_nodes_idx;
            next_nodes_idx = 0;
            depth += 1;
        }

        if !REBUILD {
            debug_assert_eq!(insert_index, 1);
        }

        bvh.nodes[0] = self.current_nodes[0];

        bvh.max_depth = DEFAULT_MAX_STACK_DEPTH.max(depth + 1);
        bvh.children_are_ordered_after_parents = !REBUILD;
    }
}

// For reference/testing
#[allow(dead_code)]
fn find_best_node_basic(index: usize, nodes: &[Bvh2Node], search_distance: usize) -> i8 {
    let mut best_node = index;
    let mut best_cost = f32::INFINITY;

    let begin = index - search_distance.min(index);
    let end = (index + search_distance + 1).min(nodes.len());

    let our_aabb = nodes[index].aabb();
    for other in begin..end {
        if other == index {
            continue;
        }
        let cost = our_aabb.union(nodes[other].aabb()).half_area();
        if cost <= best_cost {
            best_node = other;
            best_cost = cost;
        }
    }

    (best_node as i64 - index as i64) as i8
}

/// In PLOC, the number of nodes before and after the current one that are evaluated for pairing.
/// Minimum (1) has a fast path in building and still results in decent quality BVHs especially
/// when paired with a bit of reinsertion.
#[derive(Default, Clone, Copy, Debug)]
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
#[derive(Clone, Copy)]
pub struct SearchCache<const SEARCH_DISTANCE: usize>([[f32; SEARCH_DISTANCE]; SEARCH_DISTANCE]);

impl<const SEARCH_DISTANCE: usize> Default for SearchCache<SEARCH_DISTANCE> {
    fn default() -> Self {
        SearchCache([[0.0; SEARCH_DISTANCE]; SEARCH_DISTANCE])
    }
}

impl<const SEARCH_DISTANCE: usize> SearchCache<SEARCH_DISTANCE> {
    #[inline]
    fn back(&self, index: usize, other: usize) -> f32 {
        // Note: the compiler removes the bounds check due to the % SEARCH_DISTANCE
        self.0[other % SEARCH_DISTANCE][index % SEARCH_DISTANCE]
    }

    #[inline]
    fn front(&mut self, index: usize, other: usize) -> &mut f32 {
        &mut self.0[index % SEARCH_DISTANCE][other % SEARCH_DISTANCE]
    }

    #[allow(dead_code)]
    fn find_best_node_parallel(&mut self, index: usize, i: usize, nodes: &[Bvh2Node]) -> i8 {
        let mut best_node = index;
        let mut best_cost = f32::INFINITY;

        let begin = index - SEARCH_DISTANCE.min(index);
        let end = (index + SEARCH_DISTANCE + 1).min(nodes.len());

        let our_aabb = nodes[index].aabb();
        for other in begin..index {
            // When using the cache in parallel, the search is broken into chunks. This means the first
            // n = SEARCH_DISTANCE slots in the cache won't have been filled yet.
            // (TODO this could be tighter, using more of the cache within the n = SEARCH_DISTANCE range as it's filled)
            let area = if i <= SEARCH_DISTANCE {
                our_aabb.union(nodes[other].aabb()).half_area()
            } else {
                self.back(index, other)
            };

            if area <= best_cost {
                best_node = other;
                best_cost = area;
            }
        }

        ((index + 1)..end).for_each(|other| {
            let cost = our_aabb.union(nodes[other].aabb()).half_area();
            *self.front(index, other) = cost;
            if cost <= best_cost {
                best_node = other;
                best_cost = cost;
            }
        });

        (best_node as i64 - index as i64) as i8
    }

    fn find_best_node(&mut self, index: usize, nodes: &[Bvh2Node]) -> i8 {
        let mut best_node = index;
        let mut best_cost = f32::INFINITY;

        let begin = index - SEARCH_DISTANCE.min(index);
        let end = (index + SEARCH_DISTANCE + 1).min(nodes.len());

        for other in begin..index {
            let area = self.back(index, other);
            if area <= best_cost {
                best_node = other;
                best_cost = area;
            }
        }

        let our_aabb = nodes[index].aabb();
        ((index + 1)..end).for_each(|other| {
            let cost = our_aabb.union(nodes[other].aabb()).half_area();
            *self.front(index, other) = cost;
            if cost <= best_cost {
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

#[derive(Debug, Copy, Clone)]
pub enum SortPrecision {
    U128,
    U64,
}

impl SortPrecision {
    fn sort_nodes(
        &self,
        nodes: &mut [Bvh2Node],
        sorted: &mut [Bvh2Node],
        mortons_allocation: &mut [[u128; 2]],
        scale: DVec3,
        offset: DVec3,
    ) {
        match self {
            SortPrecision::U128 => {
                let mortons = cast_slice_mut(mortons_allocation);
                sort_nodes_by_morton::<Morton128>(*self, nodes, sorted, mortons, scale, offset)
            }
            SortPrecision::U64 => {
                let smaller: &mut [u128] = cast_slice_mut(mortons_allocation);
                let mortons = cast_slice_mut(&mut smaller[..nodes.len()]);
                sort_nodes_by_morton::<Morton64>(*self, nodes, sorted, mortons, scale, offset)
            }
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Morton128 {
    code: u128,
    index: u64,
    padding: u64,
}

impl RadixKey for Morton128 {
    const LEVELS: usize = 16;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        self.code.get_level(level)
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Morton64 {
    code: u64,
    index: u64,
}

impl RadixKey for Morton64 {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        self.code.get_level(level)
    }
}

trait MortonCode: RadixKey + Send + Sync + Copy {
    fn new(index: usize, center: DVec3) -> Self;
    fn index(&self) -> usize;
    fn code64(&self) -> u64;
    fn code128(&self) -> u128;
}

impl MortonCode for Morton128 {
    #[inline(always)]
    fn new(index: usize, center: DVec3) -> Self {
        Morton128 {
            index: index as u64,
            code: morton_encode_u128_unorm(center),
            padding: Default::default(),
        }
    }
    #[inline(always)]
    fn index(&self) -> usize {
        self.index as usize
    }
    #[inline(always)]
    fn code64(&self) -> u64 {
        panic!("Don't sort Morton128 using code64");
    }
    #[inline(always)]
    fn code128(&self) -> u128 {
        self.code
    }
}

impl MortonCode for Morton64 {
    #[inline(always)]
    fn new(index: usize, center: DVec3) -> Self {
        Morton64 {
            index: index as u64,
            code: morton_encode_u64_unorm(center),
        }
    }
    #[inline(always)]
    fn index(&self) -> usize {
        self.index as usize
    }
    #[inline(always)]
    fn code64(&self) -> u64 {
        self.code
    }
    #[inline(always)]
    fn code128(&self) -> u128 {
        panic!("Don't sort Morton64 using code128");
    }
}

fn sort_nodes_by_morton<M: MortonCode>(
    precision: SortPrecision,
    nodes: &mut [Bvh2Node],
    sorted_nodes: &mut [Bvh2Node],
    mortons: &mut [M],
    scale: DVec3,
    offset: DVec3,
) {
    crate::scope!("sort_nodes");
    let nodes_count = nodes.len();

    let gen_mort = |(index, (morton, leaf)): (usize, (&mut M, &Bvh2Node))| {
        let center = leaf.aabb().center().as_dvec3() * scale + offset;
        *morton = M::new(index, center);
    };

    #[cfg(feature = "parallel")]
    {
        let min_parallel = 100_000;
        if nodes_count > min_parallel {
            mortons
                .par_iter_mut()
                .zip(nodes.par_iter())
                .enumerate()
                .for_each(gen_mort);
        } else {
            mortons
                .iter_mut()
                .zip(nodes.iter())
                .enumerate()
                .for_each(gen_mort);
        }
    }
    #[cfg(not(feature = "parallel"))]
    mortons
        .iter_mut()
        .zip(nodes.iter())
        .enumerate()
        .for_each(gen_mort);

    #[cfg(feature = "parallel")]
    {
        match precision {
            SortPrecision::U128 => mortons.par_sort_unstable_by_key(|m| m.code128()),
            SortPrecision::U64 => mortons.par_sort_unstable_by_key(|m| m.code64()),
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        match nodes_count {
            0..=250_000 => match precision {
                SortPrecision::U128 => mortons.sort_unstable_by_key(|m| m.code128()),
                SortPrecision::U64 => mortons.sort_unstable_by_key(|m| m.code64()),
            },
            _ => mortons.radix_sort_unstable(),
        };
    }

    let remap = |(n, m): (&mut Bvh2Node, &M)| *n = nodes[m.index()];

    #[cfg(feature = "parallel")]
    {
        let min_parallel = 100_000;
        if nodes_count > min_parallel {
            sorted_nodes
                .par_iter_mut()
                .zip(mortons.par_iter())
                .for_each(remap)
        } else {
            sorted_nodes.iter_mut().zip(mortons.iter()).for_each(remap)
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        sorted_nodes.iter_mut().zip(mortons.iter()).for_each(remap);
    }
}
