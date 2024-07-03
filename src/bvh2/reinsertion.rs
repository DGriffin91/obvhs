// Reinsertion optimizer based on "Parallel Reinsertion for Bounding Volume Hierarchy Optimization", by D. Meister and J. Bittner:
// https://meistdan.github.io/publications/prbvh/paper.pdf
// https://jcgt.org/published/0011/04/01/paper.pdf
// Reference: https://github.com/madmann91/bvh/blob/3490634ae822e5081e41f09498fcce03bc1419e3/src/bvh/v2/reinsertion_optimizer.h

// Note: Most asserts exist to try to elide bounds checks

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rdst::{RadixKey, RadixSort};

use crate::{
    bvh2::{Bvh2, Bvh2Node},
    heapstack::HeapStack,
};

/// Restructures the BVH, optimizing node locations within the BVH hierarchy per SAH cost.
pub struct ReinsertionOptimizer<'a> {
    candidates: Vec<Candidate>,
    reinsertions: Vec<Reinsertion>,
    touched: Vec<bool>,
    parents: Vec<u32>,
    bvh: &'a mut Bvh2,
    batch_size_ratio: f32,
}

impl ReinsertionOptimizer<'_> {
    /// Restructures the BVH, optimizing node locations within the BVH hierarchy per SAH cost.
    /// batch_size_ratio: Fraction of the number of nodes to optimize per iteration.
    /// ratio_sequence: A sequence of ratios to preform reinsertion at. These are as a
    /// proportion of the batch_size_ratio. If None, the following sequence is used:
    /// (1..32).step_by(2).map(|n| 1.0 / n as f32) or
    /// 1/1, 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 1/17, 1/19, 1/21, 1/23, 1/25, 1/27, 1/29, 1/31
    pub fn run(bvh: &mut Bvh2, batch_size_ratio: f32, ratio_sequence: Option<Vec<f32>>) {
        crate::scope!("reinsertion_optimize");

        if bvh.nodes.is_empty() || bvh.nodes[0].is_leaf() || batch_size_ratio <= 0.0 {
            return;
        }
        #[cfg(feature = "parallel")]
        let parents = bvh.compute_parents_parallel();
        #[cfg(not(feature = "parallel"))]
        let parents = bvh.compute_parents();

        let cap = (bvh.nodes.len() as f32 * batch_size_ratio.min(1.0)).ceil() as usize;

        ReinsertionOptimizer {
            candidates: Vec::with_capacity(cap),
            reinsertions: Vec::with_capacity(cap),
            touched: vec![false; bvh.nodes.len()],
            parents,
            bvh,
            batch_size_ratio,
        }
        .optimize_impl(ratio_sequence);
    }

    pub fn optimize_impl(&mut self, ratio_sequence: Option<Vec<f32>>) {
        // This initially preforms reinsertion at the specified ratio, then at progressively smaller ratios,
        // focusing more reinsertion time at the top of the bvh. The original method would perform reinsertion
        // for a fixed ratio a fixed number of times.
        let ratio_sequence = ratio_sequence.unwrap_or(
            (1..32)
                .step_by(2)
                .map(|n| 1.0 / n as f32)
                .collect::<Vec<_>>(),
        );

        let mut reinsertion_stack = HeapStack::<(f32, u32)>::new_with_capacity(256); // Can't put in Self because of borrows
        ratio_sequence.iter().for_each(|ratio| {
            let batch_size =
                (((self.bvh.nodes.len() as f32 * self.batch_size_ratio) * ratio) as usize).max(1);
            let node_count = self.bvh.nodes.len().min(batch_size + 1);
            self.find_candidates(node_count);
            self.optimize_candidates(&mut reinsertion_stack, node_count - 1);
        });
    }

    /// Find potential candidates for reinsertion
    fn find_candidates(&mut self, node_count: usize) {
        // This method just takes the first node_count*2 nodes in the bvh and sorts them by their half area
        // This seemed to find candidates much faster while resulting in similar bvh traversal performance vs the original method
        // https://github.com/madmann91/bvh/blob/3490634ae822e5081e41f09498fcce03bc1419e3/src/bvh/v2/reinsertion_optimizer.h#L88
        // Taking the first node_count * 2 seemed to work nearly as well as sorting all the nodes, but builds much faster.
        self.candidates.clear();
        self.bvh
            .nodes
            .iter()
            .take(node_count * 2)
            .enumerate()
            .skip(1)
            .for_each(|(i, node)| {
                self.candidates.push(Candidate {
                    cost: node.aabb.half_area(),
                    node_id: i as u32,
                });
            });
        self.candidates.radix_sort_unstable();
    }

    #[allow(unused_variables)]
    fn optimize_candidates(&mut self, reinsertion_stack: &mut HeapStack<(f32, u32)>, count: usize) {
        self.reinsertions.clear();
        self.touched.fill(false);

        #[cfg(feature = "parallel")]
        {
            let mut reinsertions_map = (0..count)
                .into_par_iter()
                .map(|i| {
                    // TODO figure out a way to create a limited number of these just once and reuse from the rayon
                    let mut stack = HeapStack::<(f32, u32)>::new_with_capacity(256);
                    self.find_reinsertion(&mut stack, self.candidates[i].node_id as usize)
                })
                .collect::<Vec<_>>();
            reinsertions_map.drain(..).for_each(|r| {
                if r.area_diff > 0.0 {
                    self.reinsertions.push(r)
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            assert!(count <= self.candidates.len());
            (0..count).for_each(|i| {
                let r =
                    self.find_reinsertion(reinsertion_stack, self.candidates[i].node_id as usize);
                if r.area_diff > 0.0 {
                    self.reinsertions.push(r)
                }
            });
        }

        self.reinsertions
            .sort_unstable_by(|a, b| b.area_diff.partial_cmp(&a.area_diff).unwrap());

        assert!(self.reinsertions.len() <= self.touched.len());
        (0..self.reinsertions.len()).for_each(|i| {
            let reinsertion = &self.reinsertions[i];
            let conflicts = self.get_conflicts(reinsertion.from, reinsertion.to);

            if conflicts.iter().any(|&i| self.touched[i]) {
                return;
            }

            conflicts.iter().for_each(|&conflict| {
                self.touched[conflict] = true;
            });

            self.reinsert_node(reinsertion.from as usize, reinsertion.to as usize);
        });
    }

    fn find_reinsertion(&self, stack: &mut HeapStack<(f32, u32)>, node_id: usize) -> Reinsertion {
        debug_assert_ne!(node_id, 0);
        // Try to elide bounds checks
        assert!(node_id < self.bvh.nodes.len());
        assert!(node_id < self.parents.len());

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
        let node_area = self.bvh.nodes[node_id].aabb.half_area();
        let parent_area = self.bvh.nodes[self.parents[node_id] as usize]
            .aabb
            .half_area();
        let mut area_diff = parent_area;
        let mut sibling_id = Bvh2Node::get_sibling_id(node_id);
        let mut pivot_bbox = self.bvh.nodes[sibling_id].aabb;
        let parent_id = self.parents[node_id] as usize;
        let mut pivot_id = parent_id;
        let aabb = self.bvh.nodes[node_id].aabb;
        stack.clear();
        loop {
            stack.push((area_diff, sibling_id as u32));
            while !stack.is_empty() {
                let (top_area_diff, top_sibling_id) = stack.pop_fast();
                if top_area_diff - node_area <= best_reinsertion.area_diff {
                    continue;
                }

                let dst_node = &self.bvh.nodes[*top_sibling_id as usize];
                let merged_area = dst_node.aabb.union(&aabb).half_area();
                let reinsert_area = top_area_diff - merged_area;
                if reinsert_area > best_reinsertion.area_diff {
                    best_reinsertion.to = *top_sibling_id;
                    best_reinsertion.area_diff = reinsert_area;
                }

                if !dst_node.is_leaf() {
                    let child_area = reinsert_area + dst_node.aabb.half_area();
                    stack.push((child_area, dst_node.first_index));
                    stack.push((child_area, dst_node.first_index + 1));
                }
            }

            if pivot_id != parent_id {
                pivot_bbox = pivot_bbox.union(&self.bvh.nodes[sibling_id].aabb);
                area_diff += self.bvh.nodes[pivot_id].aabb.half_area() - pivot_bbox.half_area();
            }

            if pivot_id == 0 {
                break;
            }

            sibling_id = Bvh2Node::get_sibling_id(pivot_id);
            pivot_id = self.parents[pivot_id] as usize;
        }

        if best_reinsertion.to == Bvh2Node::get_sibling_id32(best_reinsertion.from)
            || best_reinsertion.to == self.parents[best_reinsertion.from as usize]
        {
            best_reinsertion = Reinsertion::default();
        }

        best_reinsertion
    }

    fn reinsert_node(&mut self, from: usize, to: usize) {
        let sibling_id = Bvh2Node::get_sibling_id(from);
        let parent_id = self.parents[from] as usize;
        let sibling_node = self.bvh.nodes[sibling_id];
        let dst_node = self.bvh.nodes[to];

        self.bvh.nodes[to].make_inner(Bvh2Node::get_left_sibling_id(from) as u32);
        self.bvh.nodes[sibling_id] = dst_node;
        self.bvh.nodes[parent_id] = sibling_node;

        if !self.bvh.nodes[sibling_id].is_leaf() {
            self.parents[self.bvh.nodes[sibling_id].first_index as usize] = sibling_id as u32;
            self.parents[self.bvh.nodes[sibling_id].first_index as usize + 1] = sibling_id as u32;
        }
        if !self.bvh.nodes[parent_id].is_leaf() {
            self.parents[self.bvh.nodes[parent_id].first_index as usize] = parent_id as u32;
            self.parents[self.bvh.nodes[parent_id].first_index as usize + 1] = parent_id as u32;
        }

        self.parents[sibling_id] = to as u32;
        self.parents[from] = to as u32;
        self.bvh.refit_from_fast(to, &self.parents);
        self.bvh.refit_from_fast(parent_id, &self.parents);
    }

    #[inline(always)]
    fn get_conflicts(&self, from: u32, to: u32) -> [usize; 5] {
        [
            to as usize,
            from as usize,
            Bvh2Node::get_sibling_id(from as usize),
            self.parents[to as usize] as usize,
            self.parents[from as usize] as usize,
        ]
    }
}

#[derive(Default, Clone, Copy)]
struct Reinsertion {
    from: u32,
    to: u32,
    area_diff: f32,
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
