// Reinsertion optimizer based on "Parallel Reinsertion for Bounding Volume Hierarchy Optimization", by D. Meister and J. Bittner:
// https://meistdan.github.io/publications/prbvh/paper.pdf
// https://jcgt.org/published/0011/04/01/paper.pdf
// Reference: https://github.com/madmann91/bvh/blob/3490634ae822e5081e41f09498fcce03bc1419e3/src/bvh/v2/reinsertion_optimizer.h

// Note: Most asserts exist to try to elide bounds checks

#[cfg(feature = "parallel")]
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use rdst::{RadixKey, RadixSort};

use crate::{
    bvh2::{Bvh2, Bvh2Node},
    fast_stack,
    faststack::FastStack,
};

use super::update_primitives_to_nodes_for_node;

/// Restructures the BVH, optimizing node locations within the BVH hierarchy per SAH cost.
#[derive(Default)]
pub struct ReinsertionOptimizer {
    candidates: Vec<Candidate>,
    reinsertions: Vec<Reinsertion>,
    touched: Vec<bool>,
    batch_size_ratio: f32,
}

impl ReinsertionOptimizer {
    /// Restructures the BVH, optimizing node locations within the BVH hierarchy per SAH cost.
    /// batch_size_ratio: Fraction of the number of nodes to optimize per iteration.
    /// ratio_sequence: A sequence of ratios to preform reinsertion at. These are as a
    /// proportion of the batch_size_ratio. If None, the following sequence is used:
    /// (1..32).step_by(2).map(|n| 1.0 / n as f32) or
    /// 1/1, 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 1/17, 1/19, 1/21, 1/23, 1/25, 1/27, 1/29, 1/31
    pub fn run(&mut self, bvh: &mut Bvh2, batch_size_ratio: f32, ratio_sequence: Option<Vec<f32>>) {
        crate::scope!("reinsertion_optimize");

        if bvh.nodes.is_empty() || bvh.nodes[0].is_leaf() || batch_size_ratio <= 0.0 {
            return;
        }

        bvh.init_parents_if_uninit();

        let cap = (bvh.nodes.len() as f32 * batch_size_ratio.min(1.0)).ceil() as usize;

        self.candidates.reserve(cap);
        self.reinsertions.reserve(cap);
        self.touched.clear();
        self.touched.resize(bvh.nodes.len(), false);
        self.batch_size_ratio = batch_size_ratio;
        self.optimize_impl(bvh, ratio_sequence);
    }

    /// Restructures the BVH, optimizing given node locations within the BVH hierarchy per SAH cost.
    ///
    /// # Arguments
    /// * `candidates` - A list of ids for nodes that need to be re-inserted.
    /// * `iterations` - The number of times reinsertion is run. Parallel reinsertion passes can result in conflicts
    ///   that potentially limit the proportion of reinsertions in a single pass.
    pub fn run_with_candidates(&mut self, bvh: &mut Bvh2, candidates: &[u32], iterations: u32) {
        crate::scope!("reinsertion_optimize_candidates");

        if bvh.nodes.is_empty() || bvh.nodes[0].is_leaf() {
            return;
        }

        bvh.init_parents_if_uninit();

        let cap = candidates.len();

        self.candidates = candidates
            .iter()
            .map(|node_id| {
                let cost = bvh.nodes[*node_id as usize].aabb().half_area();
                Candidate {
                    cost,
                    node_id: *node_id,
                }
            })
            .collect::<Vec<_>>();
        self.reinsertions.reserve(cap);
        self.touched.clear();
        self.touched.resize(bvh.nodes.len(), false);
        self.optimize_specific_candidates(bvh, iterations);
    }

    pub fn optimize_impl(&mut self, bvh: &mut Bvh2, ratio_sequence: Option<Vec<f32>>) {
        bvh.children_are_ordered_after_parents = false;
        // This initially preforms reinsertion at the specified ratio, then at progressively smaller ratios,
        // focusing more reinsertion time at the top of the bvh. The original method would perform reinsertion
        // for a fixed ratio a fixed number of times.
        let ratio_sequence = ratio_sequence.unwrap_or(
            (1..32)
                .step_by(2)
                .map(|n| 1.0 / n as f32)
                .collect::<Vec<_>>(),
        );

        ratio_sequence.iter().for_each(|ratio| {
            let batch_size =
                (((bvh.nodes.len() as f32 * self.batch_size_ratio) * ratio) as usize).max(1);
            let node_count = bvh.nodes.len().min(batch_size + 1);
            self.find_candidates(bvh, node_count);
            self.optimize_candidates(bvh, node_count - 1);
        });
    }

    pub fn optimize_specific_candidates(&mut self, bvh: &mut Bvh2, iterations: u32) {
        bvh.children_are_ordered_after_parents = false;
        for _ in 0..iterations {
            self.optimize_candidates(bvh, self.candidates.len());
        }
    }

    /// Find potential candidates for reinsertion
    fn find_candidates(&mut self, bvh: &mut Bvh2, node_count: usize) {
        // This method just takes the first node_count*2 nodes in the bvh and sorts them by their half area
        // This seemed to find candidates much faster while resulting in similar bvh traversal performance vs the original method
        // https://github.com/madmann91/bvh/blob/3490634ae822e5081e41f09498fcce03bc1419e3/src/bvh/v2/reinsertion_optimizer.h#L88
        // Taking the first node_count * 2 seemed to work nearly as well as sorting all the nodes, but builds much faster.
        self.candidates.clear();
        bvh.nodes
            .iter()
            .take(node_count * 2)
            .enumerate()
            .skip(1)
            .for_each(|(i, node)| {
                self.candidates.push(Candidate {
                    cost: node.aabb().half_area(),
                    node_id: i as u32,
                });
            });
        self.candidates.radix_sort_unstable();
    }

    #[allow(unused_variables)]
    fn optimize_candidates(&mut self, bvh: &mut Bvh2, count: usize) {
        self.touched.fill(false);

        #[cfg(feature = "parallel")]
        {
            self.reinsertions.resize(count, Default::default());
            self.reinsertions
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, reinsertion)| {
                    *reinsertion = find_reinsertion(bvh, self.candidates[i].node_id as usize)
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.reinsertions.clear();
            assert!(count <= self.candidates.len());
            (0..count).for_each(|i| {
                let r = find_reinsertion(bvh, self.candidates[i].node_id as usize);
                if r.area_diff > 0.0 {
                    self.reinsertions.push(r)
                }
            });
        }

        #[cfg(feature = "parallel")]
        self.reinsertions
            .par_sort_unstable_by(|a, b| b.area_diff.partial_cmp(&a.area_diff).unwrap());

        #[cfg(not(feature = "parallel"))]
        self.reinsertions
            .sort_unstable_by(|a, b| b.area_diff.partial_cmp(&a.area_diff).unwrap());

        assert!(self.reinsertions.len() <= self.touched.len());
        (0..self.reinsertions.len()).for_each(|i| {
            let reinsertion = &self.reinsertions[i];

            #[cfg(feature = "parallel")]
            if reinsertion.area_diff <= 0.0 {
                return;
            }

            let conflicts = self.get_conflicts(bvh, reinsertion.from, reinsertion.to);

            if conflicts.iter().any(|&i| self.touched[i]) {
                return;
            }

            conflicts.iter().for_each(|&conflict| {
                self.touched[conflict] = true;
            });

            reinsert_node(bvh, reinsertion.from as usize, reinsertion.to as usize);
        });
    }

    #[inline(always)]
    fn get_conflicts(&self, bvh: &mut Bvh2, from: u32, to: u32) -> [usize; 5] {
        [
            to as usize,
            from as usize,
            Bvh2Node::get_sibling_id(from as usize),
            // SAFETY: Caller asserts self.bvh.parents is Some outside of hot loop
            bvh.parents[to as usize] as usize,
            bvh.parents[from as usize] as usize,
        ]
    }
}

#[derive(Default, Clone, Copy)]
pub struct Reinsertion {
    pub from: u32,
    pub to: u32,
    pub area_diff: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    node_id: u32,
    cost: f32,
}

impl RadixKey for Candidate {
    const LEVELS: usize = 4;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (-self.cost).get_level(level)
    }
}

pub fn find_reinsertion(bvh: &Bvh2, node_id: usize) -> Reinsertion {
    if bvh.parents.is_empty() {
        panic!("Parents mapping required. Please run Bvh2::init_parents() before reinsert_node()")
    }

    debug_assert_ne!(node_id, 0);
    // Try to elide bounds checks
    assert!(node_id < bvh.nodes.len());
    assert!(node_id < bvh.parents.len());

    /*
     * Here is an example that explains how the cost of a reinsertion is computed. For the
     * reinsertion from A to C, in the figure below, we need to remove P1, replace it by B,
     * and create a node that holds A and C and place it where C was.
     *
     *             R
     *            / \
     *          Pn   Q1
     *          /     \
     *        ...     ...
     *        /         \
     *       P1          C
     *      / \
     *     A   B
     *
     * The resulting area *decrease* is (SA(x) means the surface area of x):
     *
     *     SA(P1) +                                                : P1 was removed
     *     SA(P2) - SA(B) +                                        : P2 now only contains B
     *     SA(P3) - SA(B U sibling(P2)) +                          : Same but for P3
     *     ... +
     *     SA(Pn) - SA(B U sibling(P2) U ... U sibling(P(n - 1)) + : Same but for Pn
     *     0 +                                                     : R does not change
     *     SA(Q1) - SA(Q1 U A) +                                   : Q1 now contains A
     *     SA(Q2) - SA(Q2 U A) +                                   : Q2 now contains A
     *     ... +
     *     -SA(A U C)                                              : For the parent of A and C
     */
    let mut best_reinsertion = Reinsertion {
        from: node_id as u32,
        to: 0,
        area_diff: 0.0,
    };
    let node_area = bvh.nodes[node_id].aabb().half_area();

    let parent_area = bvh.nodes[bvh.parents[node_id] as usize].aabb().half_area();
    let mut area_diff = parent_area;
    let mut sibling_id = Bvh2Node::get_sibling_id(node_id);
    let mut pivot_bbox = *bvh.nodes[sibling_id].aabb();
    let parent_id = bvh.parents[node_id] as usize;
    let mut pivot_id = parent_id;
    let aabb = bvh.nodes[node_id].aabb();
    let mut longest = 0;
    // TODO is it possible to push only the left pair and reduce the stack size?
    fast_stack!((f32, u32), (96, 192), bvh.max_depth * 2, stack, {
        stack.clear();
        loop {
            stack.push((area_diff, sibling_id as u32));
            while !stack.is_empty() {
                longest = stack.len().max(longest);
                let (top_area_diff, top_sibling_id) = stack.pop_fast();
                if top_area_diff - node_area <= best_reinsertion.area_diff {
                    continue;
                }

                let dst_node = &bvh.nodes[top_sibling_id as usize];
                let merged_area = dst_node.aabb().union(aabb).half_area();
                let reinsert_area = top_area_diff - merged_area;
                if reinsert_area > best_reinsertion.area_diff {
                    best_reinsertion.to = top_sibling_id;
                    best_reinsertion.area_diff = reinsert_area;
                }

                if !dst_node.is_leaf() {
                    let child_area = reinsert_area + dst_node.aabb().half_area();
                    stack.push((child_area, dst_node.first_index));
                    stack.push((child_area, dst_node.first_index + 1));
                }
            }

            if pivot_id != parent_id {
                pivot_bbox = pivot_bbox.union(bvh.nodes[sibling_id].aabb());
                area_diff += bvh.nodes[pivot_id].aabb().half_area() - pivot_bbox.half_area();
            }

            if pivot_id == 0 {
                break;
            }

            sibling_id = Bvh2Node::get_sibling_id(pivot_id);
            pivot_id = bvh.parents[pivot_id] as usize;
        }
    });

    if best_reinsertion.to == Bvh2Node::get_sibling_id32(best_reinsertion.from)
        || best_reinsertion.to == bvh.parents[best_reinsertion.from as usize]
    {
        best_reinsertion = Reinsertion::default();
    }

    best_reinsertion
}

pub fn reinsert_node(bvh: &mut Bvh2, from: usize, to: usize) {
    if bvh.parents.is_empty() {
        panic!("Parents mapping required. Please run Bvh2::init_parents() before reinsert_node()")
    }

    let sibling_id = Bvh2Node::get_sibling_id(from);
    let parent_id = bvh.parents[from] as usize;
    let sibling_node = bvh.nodes[sibling_id];
    let dst_node = bvh.nodes[to];

    bvh.nodes[to].make_inner(Bvh2Node::get_left_sibling_id(from) as u32);
    bvh.nodes[sibling_id] = dst_node;
    bvh.nodes[parent_id] = sibling_node;

    let sibling_node = &bvh.nodes[sibling_id];
    if sibling_node.is_leaf() {
        // Tell primitives where their node went.
        update_primitives_to_nodes_for_node(
            sibling_node,
            sibling_id,
            &bvh.primitive_indices,
            &mut bvh.primitives_to_nodes,
        );
    } else {
        bvh.parents[sibling_node.first_index as usize] = sibling_id as u32;
        bvh.parents[sibling_node.first_index as usize + 1] = sibling_id as u32;
    }

    let parent_node = &bvh.nodes[parent_id];
    if bvh.nodes[parent_id].is_leaf() {
        // Tell primitives where their node went.
        update_primitives_to_nodes_for_node(
            parent_node,
            parent_id,
            &bvh.primitive_indices,
            &mut bvh.primitives_to_nodes,
        );
    } else {
        bvh.parents[parent_node.first_index as usize] = parent_id as u32;
        bvh.parents[parent_node.first_index as usize + 1] = parent_id as u32;
    }

    bvh.parents[sibling_id] = to as u32;
    bvh.parents[from] = to as u32;
    bvh.refit_from_fast(to);
    bvh.refit_from_fast(parent_id);
}

#[cfg(test)]
mod tests {

    use crate::{
        ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
        test_util::geometry::demoscene,
    };

    use super::*;

    #[test]
    fn test_reinsertion() {
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
            ReinsertionOptimizer::default().run(&mut bvh, 0.25, None);
            bvh.validate(&tris, false, false);
            bvh.reorder_in_stack_traversal_order();
            ReinsertionOptimizer::default().run(&mut bvh, 0.5, None);
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
            ReinsertionOptimizer::default().run(&mut bvh, 0.25, None);
            bvh.validate(&tris, false, false);
            bvh.reorder_in_stack_traversal_order();
            ReinsertionOptimizer::default().run(&mut bvh, 0.5, None);
            bvh.validate(&tris, false, false);
        }
    }
}
