// Based on https://github.com/madmann91/bvh/blob/2fd0db62022993963a7343669275647cb073e19a/include/bvh/leaf_collapser.hpp
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::bvh2::{Bvh2, Bvh2Node};

/// Collapses leaves of the BVH according to the SAH. This optimization
/// is only helpful for bottom-up builders, as top-down builders already
/// have a termination criterion that prevents leaf creation when the SAH
/// cost does not improve.
pub fn collapse(bvh: &mut Bvh2, max_prims: u32, traversal_cost: f32) {
    crate::scope!("collapse");

    if max_prims <= 1 {
        return;
    }

    if bvh.nodes.is_empty() || bvh.nodes[0].is_leaf() {
        return;
    }

    let nodes_qty = bvh.nodes.len();

    let previously_had_parents = bvh.parents.is_some();

    bvh.init_parents();

    let mut indices_copy = Vec::new();
    let mut nodes_copy = Vec::new();

    let mut node_counts: Vec<SometimesAtomicU32> =
        (0..nodes_qty).map(|_| SometimesAtomicU32::new(1)).collect();
    let mut prim_counts: Vec<SometimesAtomicU32> =
        (0..nodes_qty).map(|_| SometimesAtomicU32::new(0)).collect();

    let node_count;

    // Bottom-up traversal to collapse leaves
    // TODO need to figure out if parallel version can have data races, if so:
    // maybe record commands in parallel, include a index, and execute them sequentially
    // also reference original impl

    assert!(bvh.parents.is_some()); // SAFETY: bottom_up_traverse assumes self.bvh.parents.is_some()
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
                let collapse_cost = node.aabb.half_area() * (total_count as f32 - traversal_cost);
                let base_cost = left.aabb.half_area() * left_count as f32
                    + right.aabb.half_area() * right_count as f32;
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

    // Prefix sums computed sequentially (TODO: parallelize)
    let mut sum = 0;
    node_counts.iter_mut().for_each(|count| {
        sum += count.get();
        count.set(sum);
    });

    sum = 0;
    prim_counts.iter_mut().for_each(|count| {
        sum += count.get();
        count.set(sum);
    });

    {
        node_count = node_counts[bvh.nodes.len() - 1].get();
        if prim_counts[0].get() > 0 {
            // This means the root node has become a leaf.
            // We avoid copying the data and just swap the old prim array with the new one.
            bvh.nodes[0].first_index = 0;
            bvh.nodes[0].prim_count = prim_counts[0].get();
            std::mem::swap(&mut bvh.primitive_indices, &mut indices_copy);
            std::mem::swap(&mut bvh.nodes, &mut nodes_copy);
        } else {
            nodes_copy = vec![Default::default(); node_count as usize];
            indices_copy =
                vec![Default::default(); prim_counts[bvh.nodes.len() - 1].get() as usize];
            nodes_copy[0] = bvh.nodes[0];
            nodes_copy[0].first_index = node_counts[nodes_copy[0].first_index as usize - 1].get();
        }
    }

    // TODO Parallelize:
    {
        for i in 1..bvh.nodes.len() {
            let node_id = node_counts[i - 1].get() as usize;
            if node_id == node_counts[i].get() as usize {
                continue;
            }

            nodes_copy[node_id] = bvh.nodes[i];
            let mut first_prim = prim_counts[i - 1].get();
            if first_prim != prim_counts[i].get() {
                nodes_copy[node_id].prim_count = prim_counts[i].get() - first_prim;
                nodes_copy[node_id].first_index = first_prim;

                // Top-down traversal to store the prims contained in this subtree.

                if true {
                    let mut j = i;
                    loop {
                        let node = bvh.nodes[j];
                        if node.is_leaf() {
                            for n in 0..node.prim_count {
                                indices_copy[(first_prim + n) as usize] =
                                    bvh.primitive_indices[(node.first_index + n) as usize];
                            }

                            first_prim += node.prim_count;
                            while !Bvh2Node::is_left_sibling(j) && j != i {
                                // SAFETY: Caller asserts self.bvh.parents is Some outside of hot loop
                                j = unsafe { bvh.parents.as_ref().unwrap_unchecked() }[j] as usize;
                            }
                            if j == i {
                                break;
                            }
                            j = Bvh2Node::get_sibling_id(j);
                        } else {
                            j = node.first_index as usize;
                        }
                    }
                } else {
                    // -------------------------
                    // Alternate method (slower)
                    // -------------------------
                    let mut stack = Vec::new();
                    stack.push(i);
                    while let Some(current_node_index) = stack.pop() {
                        let node = &bvh.nodes[current_node_index];

                        if node.is_leaf() {
                            for n in 0..node.prim_count {
                                indices_copy[(first_prim + n) as usize] =
                                    bvh.primitive_indices[(node.first_index + n) as usize];
                            }
                            first_prim += node.prim_count;
                        } else {
                            stack.push(node.first_index as usize);
                            stack.push((node.first_index + 1) as usize);
                        }
                    }
                    // -------------------------
                }
            } else {
                let first_child = &mut nodes_copy[node_id].first_index;
                *first_child = node_counts[*first_child as usize - 1].get();
            }
        }
    }

    std::mem::swap(&mut bvh.nodes, &mut nodes_copy);
    std::mem::swap(&mut bvh.primitive_indices, &mut indices_copy);

    if previously_had_parents {
        // If we had parents already computed before collapse we need to recompute them now, if not, skip the extra computation
        bvh.recompute_parents();
    }
}

// Based on https://github.com/madmann91/bvh/blob/2fd0db62022993963a7343669275647cb073e19a/include/bvh/bottom_up_algorithm.hpp
#[cfg(not(feature = "parallel"))]
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

    // Iterate through all nodes starting from 1, since node 0 is assumed to be the root
    (1..bvh.nodes.len()).for_each(|i| {
        // Only process leaves
        if bvh.nodes[i].is_leaf() {
            process_node(true, i);

            // Process inner nodes on the path from that leaf up to the root
            let mut j = i;
            while j != 0 {
                // SAFETY: Caller asserts self.bvh.parents is Some outside of hot loop
                j = unsafe { bvh.parents.as_ref().unwrap_unchecked() }[j] as usize;

                process_node(false, j);
            }
        }
    });
}

#[cfg(feature = "parallel")]
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

    // Iterate through all nodes starting from 1, since node 0 is assumed to be the root
    (1..bvh.nodes.len()).into_par_iter().for_each(|i| {
        // Only process leaves
        if bvh.nodes[i].is_leaf() {
            process_node(true, i);

            // Process inner nodes on the path from that leaf up to the root
            let mut j = i as usize;
            while j != 0 {
                // SAFETY: Caller asserts self.bvh.parents is Some outside of hot loop
                j = unsafe { bvh.parents.as_ref().unwrap_unchecked() }[j] as usize;

                process_node(false, j);
            }
        }
    });
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
