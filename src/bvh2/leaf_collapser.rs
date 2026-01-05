use bytemuck::zeroed_vec;

// Based on https://github.com/madmann91/bvh/blob/2fd0db62022993963a7343669275647cb073e19a/include/bvh/leaf_collapser.hpp
#[cfg(feature = "parallel")]
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::bvh2::{Bvh2, Bvh2Node};

/// Collapses leaves of the BVH according to the SAH. This optimization
/// is only helpful for bottom-up builders, as top-down builders already
/// have a termination criterion that prevents leaf creation when the SAH
/// cost does not improve.
pub fn collapse(bvh: &mut Bvh2, max_prims: u32, traversal_cost: f32) {
    crate::scope!("collapse");
    let nodes_qty = bvh.nodes.len();

    if max_prims <= 1 || nodes_qty as u32 <= max_prims * 2 + 1 {
        return;
    }

    if !bvh.primitive_indices.is_empty() && bvh.primitive_indices.len() as u32 <= max_prims {
        return;
    }

    if bvh.nodes.is_empty() || bvh.nodes[0].is_leaf() {
        return;
    }

    let previously_had_parents = !bvh.parents.is_empty();

    bvh.init_parents_if_uninit();

    let mut node_counts = vec![1u32; nodes_qty];
    let mut prim_counts = vec![0u32; nodes_qty];
    let node_count;

    {
        let node_counts = as_slice_of_sometimes_atomic_u32(&mut node_counts);
        let prim_counts = as_slice_of_sometimes_atomic_u32(&mut prim_counts);

        // Bottom-up traversal to collapse leaves
        bottom_up_traverse(bvh, |leaf, i| {
            if leaf {
                prim_counts[i].set(bvh.nodes[i].prim_count);
            } else {
                let node = &bvh.nodes[i];
                debug_assert!(!node.is_leaf());
                let first_child = node.first_index as usize;

                let left_count = prim_counts[first_child].get();
                let right_count = prim_counts[first_child + 1].get();
                let total_count = left_count + right_count;

                // Compute the cost of collapsing this node when both children are leaves
                if left_count > 0 && right_count > 0 && total_count <= max_prims {
                    let left = bvh.nodes[first_child];
                    let right = bvh.nodes[first_child + 1];
                    let collapse_cost =
                        node.aabb().half_area() * (total_count as f32 - traversal_cost);
                    let base_cost = left.aabb().half_area() * left_count as f32
                        + right.aabb().half_area() * right_count as f32;
                    let both_have_same_prim =
                        (left.first_index == right.first_index) && total_count == 2;

                    // Collapse them if cost of the collapsed node is lower, or both children contain the same primitive (as a result of splits)
                    if collapse_cost <= base_cost || both_have_same_prim {
                        //if both_have_same_prim { 1 } else { total_count }; // TODO, Reduce total count (was showing artifacts)
                        prim_counts[i].set(total_count);
                        prim_counts[first_child].set(0);
                        prim_counts[first_child + 1].set(0);
                        node_counts[first_child].set(0);
                        node_counts[first_child + 1].set(0);
                    }
                }
            }
        });
    }

    #[cfg(feature = "parallel")]
    {
        parallel_prefix_sum(&mut node_counts);
        parallel_prefix_sum(&mut prim_counts);
    }
    #[cfg(not(feature = "parallel"))]
    {
        prefix_sum(&mut node_counts);
        prefix_sum(&mut prim_counts);
    }

    let mut indices_copy = Vec::new();
    let mut nodes_copy = Vec::new();
    {
        node_count = node_counts[bvh.nodes.len() - 1];
        if prim_counts[0] > 0 {
            // This means the root node has become a leaf.
            // We avoid copying the data and just swap the old prim array with the new one.
            bvh.nodes[0].first_index = 0;
            bvh.nodes[0].prim_count = prim_counts[0];
            std::mem::swap(&mut bvh.primitive_indices, &mut indices_copy);
            std::mem::swap(&mut bvh.nodes, &mut nodes_copy);
        } else {
            nodes_copy = zeroed_vec(node_count as usize);
            indices_copy = zeroed_vec(prim_counts[bvh.nodes.len() - 1] as usize);
            nodes_copy[0] = bvh.nodes[0];
            nodes_copy[0].first_index = node_counts[nodes_copy[0].first_index as usize - 1];
        }
    }

    {
        let indices_copy = as_slice_of_sometimes_atomic_u32(&mut indices_copy);

        #[cfg(feature = "parallel")]
        let mut needs_traversal = Vec::with_capacity(bvh.nodes.len().div_ceil(4));

        #[allow(unused_mut)]
        let mut top_down_traverse = |i| {
            // Top-down traversal to store the prims contained in this subtree.
            #[allow(clippy::unnecessary_cast)]
            let i = i as usize;
            let mut first_prim = prim_counts[i - 1];
            let mut j = i;
            loop {
                let node = bvh.nodes[j];
                if node.is_leaf() {
                    for n in 0..node.prim_count {
                        indices_copy[(first_prim + n) as usize]
                            .set(bvh.primitive_indices[(node.first_index + n) as usize]);
                    }

                    first_prim += node.prim_count;
                    while !Bvh2Node::is_left_sibling(j) && j != i {
                        j = bvh.parents[j] as usize;
                    }
                    if j == i {
                        break;
                    }
                    j = Bvh2Node::get_sibling_id(j);
                } else {
                    j = node.first_index as usize;
                }
            }
        };

        (1..bvh.nodes.len()).for_each(|i| {
            let node_id = node_counts[i - 1] as usize;
            if node_id == node_counts[i] as usize {
                return;
            }
            nodes_copy[node_id] = bvh.nodes[i];
            let first_prim = prim_counts[i - 1];
            if first_prim == prim_counts[i] {
                let first_child = &mut nodes_copy[node_id].first_index;
                *first_child = node_counts[*first_child as usize - 1];
            } else {
                nodes_copy[node_id].prim_count = prim_counts[i] - first_prim;
                nodes_copy[node_id].first_index = first_prim;
                #[cfg(feature = "parallel")]
                needs_traversal.push(i as u32);
                #[cfg(not(feature = "parallel"))]
                top_down_traverse(i);
            }
        });

        #[cfg(feature = "parallel")]
        needs_traversal.into_par_iter().for_each(top_down_traverse);
    }

    std::mem::swap(&mut bvh.nodes, &mut nodes_copy);
    std::mem::swap(&mut bvh.primitive_indices, &mut indices_copy);

    if previously_had_parents {
        // If we had parents already computed before collapse we need to recompute them now
        // TODO perf there might be a way to update this during collapse
        bvh.update_parents();
    } else {
        // If not, skip the extra computation
        bvh.parents.clear();
    }
    if !bvh.primitives_to_nodes.is_empty() {
        // If primitives_to_nodes already existed we need to make sure it remains valid.
        // TODO perf there might be a way to update this during collapse
        bvh.update_primitives_to_nodes();
    }
}

// Based on https://github.com/madmann91/bvh/blob/2fd0db62022993963a7343669275647cb073e19a/include/bvh/bottom_up_algorithm.hpp
#[cfg(not(feature = "parallel"))]
/// Caller must make sure Bvh2::parents is initialized
fn bottom_up_traverse<F>(
    bvh: &Bvh2,
    mut process_node: F, // True is for leaf
) where
    F: FnMut(bool, usize),
{
    // Special case if the BVH is just a leaf
    if bvh.nodes.len() == 1 {
        process_node(true, 0);
        return;
    }

    let mut flags: Vec<u8> = zeroed_vec(bvh.nodes.len());

    // Iterate through all nodes starting from 1, since node 0 is assumed to be the root
    (1..bvh.nodes.len()).for_each(|i| {
        // Always start at leaf
        if bvh.nodes[i].is_leaf() {
            process_node(true, i);

            // Process inner nodes on the path from that leaf up to the root
            let mut j = i;
            while j != 0 {
                j = bvh.parents[j] as usize;

                let flag = &mut flags[j];

                // Make sure that the children of this inner node have been processed
                let previous_flag = *flag;
                *flag = previous_flag.saturating_add(1);
                if previous_flag != 1 {
                    break;
                }
                *flag = 0;

                process_node(false, j);
            }
        }
    });
}

// Based on https://github.com/madmann91/bvh/blob/2fd0db62022993963a7343669275647cb073e19a/include/bvh/bottom_up_algorithm.hpp
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
// Paths from leaf nodes to the root are processed in parallel. Each thread starts from one leaf node and walks up the
// tree using parent pointers. We track how many threads have visited each internal node using atomic counters—the first
// thread terminates immediately while the second one gets to process the node. This way, each node is processed by
// exactly one thread, which leads to O(n) time complexity.
#[cfg(feature = "parallel")]
/// Caller must make sure Bvh2::parents is initialized
fn bottom_up_traverse<F>(
    bvh: &Bvh2,
    process_node: F, // True is for leaf
) where
    F: Fn(bool, usize) + Sync + Send,
{
    // Special case if the BVH is just a leaf

    if bvh.nodes.len() == 1 {
        process_node(true, 0);
        return;
    }

    // Compiles down to just alloc_zeroed https://users.rust-lang.org/t/create-vector-of-atomicusize-etc/121695/5
    let flags = vec![0u32; bvh.nodes.len()]
        .into_iter()
        .map(AtomicU32::new)
        .collect::<Vec<_>>();

    // Iterate through all nodes starting from 1, since node 0 is assumed to be the root
    (1..bvh.nodes.len()).into_par_iter().for_each(|i| {
        // Always start at leaf
        if bvh.nodes[i].is_leaf() {
            process_node(true, i);

            // Process inner nodes on the path from that leaf up to the root
            let mut j = i;
            while j != 0 {
                j = bvh.parents[j] as usize;

                let flag = &flags[j];

                // Make sure that the children of this inner node have been processed
                if flag.fetch_add(1, Ordering::SeqCst) != 1 {
                    break;
                }
                flag.store(0, Ordering::SeqCst);

                process_node(false, j);
            }
        }
    });
}

#[cfg(feature = "parallel")]
fn parallel_prefix_sum<T>(data: &mut [T])
where
    T: std::ops::Add + std::ops::AddAssign + Send + Default + Clone + Copy,
{
    // Split into chunks
    let chunk_size = 1.max(data.len().div_ceil(rayon::current_num_threads()));
    let chunks = data.par_chunks_mut(chunk_size);
    let mut partial_sums: Vec<T> = vec![Default::default(); chunks.len()];

    // Compute local prefix sum in parallel
    chunks
        .zip(partial_sums.par_iter_mut())
        .for_each(|(chunk, partial_sum)| *partial_sum = prefix_sum(chunk));

    // Compute partial sums
    prefix_sum(&mut partial_sums);

    // Apply partial sums
    data.par_chunks_mut(chunk_size)
        .skip(1)
        .zip(partial_sums)
        .for_each(|(chunk, partial_sum)| chunk.iter_mut().for_each(move |n| *n += partial_sum));
}

#[inline]
fn prefix_sum<T>(data: &mut [T]) -> T
where
    T: std::ops::Add + std::ops::AddAssign + Send + Default + Clone + Copy,
{
    let mut sum: T = Default::default();
    data.iter_mut().for_each(|count| {
        sum += *count;
        *count = sum;
    });
    sum
}

pub struct SometimesAtomicU32 {
    #[cfg(feature = "parallel")]
    pub value: AtomicU32,
    #[cfg(not(feature = "parallel"))]
    pub value: u32,
}

impl SometimesAtomicU32 {
    #[inline]
    pub fn new(value: u32) -> SometimesAtomicU32 {
        #[cfg(feature = "parallel")]
        {
            SometimesAtomicU32 {
                value: AtomicU32::new(value),
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            SometimesAtomicU32 { value }
        }
    }

    #[inline]
    #[cfg(feature = "parallel")]
    pub fn set(&self, value: u32) {
        self.value.store(value, Ordering::SeqCst);
        #[cfg(not(feature = "parallel"))]
        {
            self.value = value;
        }
    }

    #[inline]
    #[cfg(not(feature = "parallel"))]
    pub fn set(&mut self, value: u32) {
        self.value = value;
    }

    #[inline]
    pub fn get(&self) -> u32 {
        #[cfg(feature = "parallel")]
        {
            self.value.load(Ordering::SeqCst)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.value
        }
    }
}

#[inline]
fn as_slice_of_sometimes_atomic_u32(slice: &mut [u32]) -> &mut [SometimesAtomicU32] {
    assert_eq!(size_of::<SometimesAtomicU32>(), size_of::<u32>());
    assert_eq!(align_of::<SometimesAtomicU32>(), align_of::<u32>());
    let atomic_slice: &mut [SometimesAtomicU32] = unsafe {
        core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut SometimesAtomicU32, slice.len())
    };
    // Alternatively:
    //let slice: &mut [SometimesAtomicU32] = unsafe { &mut *((slice.as_mut_slice() as *mut [u32]) as *mut [SometimesAtomicU32]) };
    atomic_slice
}

#[cfg(test)]
mod tests {

    use crate::{
        ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
        test_util::geometry::demoscene,
    };

    use super::*;

    #[test]
    fn test_collapse() {
        let tris = demoscene(32, 0);
        let mut aabbs = Vec::with_capacity(tris.len());
        let mut indices = Vec::with_capacity(tris.len());
        for (i, primitive) in tris.iter().enumerate() {
            indices.push(i as u32);
            aabbs.push(primitive.aabb());
        }
        {
            // Test without init_primitives_to_nodes & init_parents
            let mut bvh = PlocBuilder::new().build(
                PlocSearchDistance::VeryLow,
                &aabbs,
                indices.clone(),
                SortPrecision::U64,
                1,
            );
            bvh.validate(&tris, false, false);
            collapse(&mut bvh, 8, 1.0);
            bvh.validate(&tris, false, false);
        }
        {
            // Test with init_primitives_to_nodes & init_parents
            let mut bvh = PlocBuilder::new().build(
                PlocSearchDistance::VeryLow,
                &aabbs,
                indices,
                SortPrecision::U64,
                1,
            );
            bvh.validate(&tris, false, false);
            bvh.init_primitives_to_nodes_if_uninit();
            bvh.init_parents_if_uninit();
            bvh.validate(&tris, false, false);
            collapse(&mut bvh, 8, 1.0);
            bvh.validate(&tris, false, false);
        }
    }
}
