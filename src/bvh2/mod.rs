//! A binary BVH

pub mod builder;
pub mod leaf_collapser;
pub mod reinsertion;

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicU32, Ordering};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use bytemuck::{Pod, Zeroable};

#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    aabb::Aabb,
    heapstack::HeapStack,
    ray::{Ray, RayHit},
    Boundable,
};

/// A node in the Bvh2, can be an inner node or leaf.
#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Bvh2Node {
    /// The bounding box for the primitive(s) contain in this node
    pub aabb: Aabb,
    /// Number of primitives contained in this node.
    /// If prim_count is 0, this is a inner node.
    /// If prim_count > 0 this node is a leaf node.
    pub prim_count: u32,
    /// The index of the first child Aabb or primitive.
    /// If this node is an inner node the first child will be at `nodes[first_index]`, and the second at `nodes[first_index + 1]`.
    /// If this node is a leaf node the first index typically indexes into a primitive_indices list that contains the actual index of the primitive.
    /// The reason for this mapping is that if multiple primitives are contained in this node, they need to have their indices layed out contiguously.
    /// To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub first_index: u32,
}

unsafe impl Pod for Bvh2Node {}
unsafe impl Zeroable for Bvh2Node {}

impl Bvh2Node {
    #[inline(always)]
    pub fn new(aabb: Aabb, prim_count: u32, first_index: u32) -> Self {
        Self {
            aabb,
            prim_count,
            first_index,
        }
    }

    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.prim_count != 0
    }

    #[inline(always)]
    pub fn is_left_sibling(node_id: usize) -> bool {
        node_id % 2 == 1
    }
    #[inline(always)]
    pub fn get_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id + 1
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_left_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_right_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id + 1
        } else {
            node_id
        }
    }

    #[inline(always)]
    pub fn is_left_sibling32(node_id: u32) -> bool {
        node_id % 2 == 1
    }
    #[inline(always)]
    pub fn get_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id + 1
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_left_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_right_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id + 1
        } else {
            node_id
        }
    }
    #[inline(always)]
    pub fn make_inner(&mut self, first_index: u32) {
        self.prim_count = 0;
        self.first_index = first_index;
    }
}

/// Holds Ray traversal state to allow for dynamic traversal (yield on hit)
pub struct RayTraversal {
    pub stack: HeapStack<u32>,
    pub ray: Ray,
    pub current_primitive_index: u32,
    pub primitive_count: u32,
}

impl RayTraversal {
    #[inline(always)]
    /// Reinitialize traversal state with new ray.
    pub fn reinit(&mut self, ray: Ray) {
        self.stack.clear();
        self.stack.push(0);
        self.primitive_count = 0;
        self.current_primitive_index = 0;
        self.ray = ray;
    }
}

/// A binary BVH
#[derive(Clone)]
pub struct Bvh2 {
    /// List of nodes contained in this bvh. first_index in Bvh2Node indexes into this list.
    pub nodes: Vec<Bvh2Node>,
    /// Mapping from bvh primitive indices to original input indices
    /// The reason for this mapping is that if multiple primitives are contained in this node, they need to have their indices layed out contiguously.
    /// To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub primitive_indices: Vec<u32>,
    /// Maximum bvh hierarchy depth. Used to determine stack depth for cpu bvh2 traversal.
    /// Stack defaults to 96 or the max depth during initial ploc building, whichever is larger. This may be larger than
    /// needed depending on what post processing steps (like collapse, reinsertion, etc...), but the cost of
    /// recalculating it may not be worth it so it is not done automatically.
    pub max_depth: usize,
}
pub const DEFAULT_MAX_STACK_DEPTH: usize = 96;

impl Default for Bvh2 {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            primitive_indices: Default::default(),
            max_depth: DEFAULT_MAX_STACK_DEPTH,
        }
    }
}

impl Bvh2 {
    #[inline(always)]
    pub fn new_ray_traversal(&self, ray: Ray) -> RayTraversal {
        let mut stack = HeapStack::new_with_capacity(self.max_depth);
        if !self.nodes.is_empty() {
            stack.push(0);
        }
        RayTraversal {
            stack,
            ray,
            current_primitive_index: 0,
            primitive_count: 0,
        }
    }

    /// Traverse the bvh for a given `Ray`. Returns the closest intersected primitive.
    /// `primitives` should be the list of primitives used to generate the bvh reordered per Bvh2::primitive_indices.
    /// To avoid needing to reorder the primitives at the cost of one layer of indirection, see traverse_indirect.
    ///
    /// # Arguments
    /// * `ray` - The ray to be tested for intersection.
    /// * `hit` - As traverse_dynamic intersects primitives, it will update `hit` with the closest.
    /// * `intersection_fn` - should take the given ray and primitive index and return the distance to the intersection, if any.
    ///   Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    ///   Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    ///   original primitives per primitive_indices.
    #[inline(always)]
    pub fn ray_traverse<F: FnMut(&Ray, usize) -> f32>(
        &self,
        ray: Ray,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        let mut state = self.new_ray_traversal(ray);
        while self.ray_traverse_dynamic(&mut state, hit, &mut intersection_fn) {}
        hit.t < ray.tmax // Note this is valid since traverse_with_stack does not mutate the ray
    }

    /// Traverse the BVH
    /// Yields at every primitive hit, returning true.
    /// Returns false when no hit is found.
    /// For basic miss test, just run until the first time it yields true.
    /// For closest hit run until it returns false and check hit.t < ray.tmax to see if it hit something
    /// For transparency, you want to hit every primitive in the ray's path, keeping track of the closest opaque hit.
    ///     and then manually setting ray.tmax to that closest opaque hit at each iteration.
    ///
    /// # Arguments
    /// * `state` - Holds the current traversal state. Allows traverse_dynamic to yield.
    /// * `hit` - As traverse_dynamic intersects primitives, it will update `hit` with the closest.
    /// * `intersection_fn` - should take the given ray and primitive index and return the distance to the intersection, if any.
    ///   Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    ///   Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    ///   original primitives per primitive_indices.
    #[inline(always)]
    pub fn ray_traverse_dynamic<F: FnMut(&Ray, usize) -> f32>(
        &self,
        state: &mut RayTraversal,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        loop {
            while state.primitive_count > 0 {
                let primitive_id = state.current_primitive_index;
                state.current_primitive_index += 1;
                state.primitive_count -= 1;
                let t = intersection_fn(&state.ray, primitive_id as usize);
                if t < state.ray.tmax {
                    hit.primitive_id = primitive_id;
                    hit.t = t;
                    state.ray.tmax = t;
                    // Yield when we hit a primitive
                    return true;
                }
            }
            if let Some(current_node_index) = state.stack.pop() {
                let node = &self.nodes[*current_node_index as usize];
                if node.aabb.intersect_ray(&state.ray) >= state.ray.tmax {
                    continue;
                }

                if node.is_leaf() {
                    state.primitive_count = node.prim_count;
                    state.current_primitive_index = node.first_index;
                } else {
                    state.stack.push(node.first_index);
                    state.stack.push(node.first_index + 1);
                }
            } else {
                // Returns false when there are no more primitives to test.
                // This doesn't mean we never hit one along the way though. (and yielded then)
                return false;
            }
        }
    }

    /// Recursively traverse the bvh for a given `Ray`.
    /// On completion, `indices` will contain a list of the intersected leaf nodes.
    /// This method is slower than stack traversal and only exists as a reference.
    /// This method does not check if the primitive was intersected, only the leaf node.
    pub fn ray_traverse_recursive(&self, ray: &Ray, node_index: usize, indices: &mut Vec<usize>) {
        let node = &self.nodes[node_index];

        if node.is_leaf() {
            let primitive_id = node.first_index as usize;
            indices.push(primitive_id);
        } else if node.aabb.intersect_ray(ray) < f32::INFINITY {
            self.ray_traverse_recursive(ray, node.first_index as usize, indices);
            self.ray_traverse_recursive(ray, node.first_index as usize + 1, indices);
        }
    }

    /// Traverse the BVH with an Aabb. fn `eval` is called for nodes that intersect `aabb`
    /// The bvh (self) and the current node index is passed into fn `eval`
    /// Note each node may have multiple primitives. `node.first_index` is the index of the first primitive.
    /// `node.prim_count` is the quantity of primitives contained in the given node.
    /// Return false from eval to halt traversal
    pub fn aabb_traverse<F: FnMut(&Self, u32) -> bool>(&self, aabb: Aabb, mut eval: F) {
        let mut stack = HeapStack::new_with_capacity(self.max_depth);
        stack.push(0);
        while let Some(current_node_index) = stack.pop() {
            let node = &self.nodes[*current_node_index as usize];
            if !node.aabb.intersect_aabb(&aabb) {
                continue;
            }

            if node.is_leaf() {
                if !eval(self, *current_node_index) {
                    return;
                }
            } else {
                stack.push(node.first_index);
                stack.push(node.first_index + 1);
            }
        }
    }

    // Order node array in stack traversal order. (Doesn't seem to really speed up traversal)
    pub fn reorder_in_stack_traversal_order(&mut self) {
        let mut new_nodes: Vec<Bvh2Node> = Vec::with_capacity(self.nodes.len());
        let mut mapping = vec![0; self.nodes.len()]; // Map from where n node used to be to where it is now
        let mut stack = Vec::new();
        stack.push(1);
        new_nodes.push(self.nodes[0]);
        mapping[0] = 0;
        while let Some(current_node_index) = stack.pop() {
            let node_a = &self.nodes[current_node_index as usize];
            let node_b = &self.nodes[current_node_index as usize + 1];
            if !node_a.is_leaf() {
                stack.push(node_a.first_index);
            }
            if !node_b.is_leaf() {
                stack.push(node_b.first_index);
            }
            let new_node_idx = new_nodes.len() as u32;
            mapping[current_node_index as usize] = new_node_idx;
            mapping[current_node_index as usize + 1] = new_node_idx + 1;
            new_nodes.push(*node_a);
            new_nodes.push(*node_b);
        }
        for n in &mut new_nodes {
            if !n.is_leaf() {
                n.first_index = mapping[n.first_index as usize];
            }
        }
        self.nodes = new_nodes;
    }

    /// Returns the list of parents where `parent_index = parents[node_index]`
    pub fn compute_parents(&self) -> Vec<u32> {
        let mut parents = vec![0; self.nodes.len()];
        parents[0] = 0;
        self.nodes.iter().enumerate().for_each(|(i, node)| {
            if !node.is_leaf() {
                parents[node.first_index as usize] = i as u32;
                parents[node.first_index as usize + 1] = i as u32;
            }
        });
        parents
    }

    /// Refit the BVH working up the tree from this node, ignoring leaves. (TODO add a version that checks leaves)
    /// This recomputes the Aabbs for all the parents of the given node index.
    /// This can only be used to refit when a single node has changed or moved.
    pub fn refit_from(&mut self, mut index: usize, parents: &[u32]) {
        loop {
            let node = &self.nodes[index];
            if !node.is_leaf() {
                let first_child_bbox = self.nodes[node.first_index as usize].aabb;
                let second_child_bbox = self.nodes[node.first_index as usize + 1].aabb;
                self.nodes[index].aabb = first_child_bbox.union(&second_child_bbox);
            }
            if index == 0 {
                break;
            }
            index = parents[index] as usize;
        }
    }

    /// Refit the BVH working up the tree from this node, ignoring leaves.
    /// This recomputes the Aabbs for the parents of the given node index.
    /// Halts if the parents are the same size. Panics in debug if some parents still needed to be resized.
    /// This can only be used to refit when a single node has changed or moved.
    pub fn refit_from_fast(&mut self, mut index: usize, parents: &[u32]) {
        let mut same_count = 0;
        loop {
            let node = &self.nodes[index];
            if !node.is_leaf() {
                let first_child_bbox = self.nodes[node.first_index as usize].aabb;
                let second_child_bbox = self.nodes[node.first_index as usize + 1].aabb;
                let new_aabb = first_child_bbox.union(&second_child_bbox);
                let node = &mut self.nodes[index].aabb;
                if node == &new_aabb {
                    same_count += 1;
                    #[cfg(not(debug_assertions))]
                    if same_count == 2 {
                        return;
                    }
                } else {
                    debug_assert!(same_count < 2, "Some parents still needed refitting. Unideal fitting is occurring somewhere.");
                }
                *node = new_aabb;
            }
            if index == 0 {
                break;
            }
            index = parents[index] as usize;
        }
    }

    /// Direct layout: The primitives are already laid out in bvh.primitive_indices order.
    pub fn validate<T: Boundable>(
        &self,
        primitives: &[T],
        direct_layout: bool,
        splits: bool,
    ) -> Bvh2ValidationResult {
        if !splits {
            // Could still check this if duplicated were removed from self.primitive_indices first
            assert_eq!(self.primitive_indices.len(), primitives.len());
        }
        let mut result = Bvh2ValidationResult {
            splits,
            direct_layout,
            ..Default::default()
        };

        if !self.nodes.is_empty() {
            self.validate_impl::<T>(primitives, &mut result, 0, 0, 0);
        }
        assert_eq!(result.discovered_nodes.len(), self.nodes.len());
        assert_eq!(result.node_count, self.nodes.len());
        assert_eq!(
            result.discovered_primitives.len(),
            self.primitive_indices.len()
        );
        assert_eq!(result.prim_count, self.primitive_indices.len());
        assert!(
            result.max_depth < self.max_depth as u32,
            "result.max_depth ({}) must be less than self.max_depth ({})",
            result.max_depth,
            self.max_depth as u32
        );
        if result.max_depth > DEFAULT_MAX_STACK_DEPTH as u32 {
            log::warn!("bvh depth is: {}, a depth beyond {} may be indicative of something pathological in the scene (like thousands of instances perfectly overlapping geometry) that will result in a BVH that is very slow to traverse.", result.max_depth, DEFAULT_MAX_STACK_DEPTH);
        }

        result
    }

    pub fn validate_impl<T: Boundable>(
        &self,
        primitives: &[T],
        result: &mut Bvh2ValidationResult,
        node_index: u32,
        parent_index: u32,
        current_depth: u32,
    ) {
        result.max_depth = result.max_depth.max(current_depth);
        let parent_aabb = self.nodes[parent_index as usize].aabb;
        result.discovered_nodes.insert(node_index);
        let node = &self.nodes[node_index as usize];
        result.node_count += 1;

        if let Some(count) = result.nodes_at_depth.get(&current_depth) {
            result.nodes_at_depth.insert(current_depth, count + 1);
        } else {
            result.nodes_at_depth.insert(current_depth, 1);
        }

        assert!(
            node.aabb.min.cmpge(parent_aabb.min).all()
                && node.aabb.max.cmple(parent_aabb.max).all(),
            "Child {} does not fit in parent {}:\nchild:  {:?}\nparent: {:?}",
            node_index,
            parent_index,
            node.aabb,
            parent_aabb
        );

        if node.is_leaf() {
            result.leaf_count += 1;
            if let Some(count) = result.leaves_at_depth.get(&current_depth) {
                result.leaves_at_depth.insert(current_depth, count + 1);
            } else {
                result.leaves_at_depth.insert(current_depth, 1);
            }
            for i in 0..node.prim_count {
                result.prim_count += 1;
                let mut prim_index = (node.first_index + i) as usize;
                result.discovered_primitives.insert(prim_index as u32);
                // If using splits, primitives will extend outside the leaf in some cases.
                if !result.splits {
                    if !result.direct_layout {
                        prim_index = self.primitive_indices[prim_index] as usize
                    }
                    let prim_aabb = primitives[prim_index].aabb();
                    assert!(
                        prim_aabb.min.cmpge(node.aabb.min).all()
                            && prim_aabb.max.cmple(node.aabb.max).all(),
                        "Primitive {} does not fit in parent {}:\nprimitive: {:?}\nparent:    {:?}",
                        prim_index,
                        parent_index,
                        prim_aabb,
                        node.aabb
                    );
                }
            }
        } else {
            self.validate_impl::<T>(
                primitives,
                result,
                node.first_index,
                parent_index,
                current_depth + 1,
            );
            self.validate_impl::<T>(
                primitives,
                result,
                node.first_index + 1,
                parent_index,
                current_depth + 1,
            );
        }
    }

    // Seems around 80% faster than compute_parents.
    // TODO is there a better way to parallelize?
    #[cfg(feature = "parallel")]
    pub fn compute_parents_parallel(&self) -> Vec<u32> {
        let parents: Vec<AtomicU32> = (0..self.nodes.len()).map(|_| AtomicU32::new(0)).collect();

        parents[0].store(0, Ordering::Relaxed);

        self.nodes.par_iter().enumerate().for_each(|(i, node)| {
            if !node.is_leaf() {
                parents[node.first_index as usize].store(i as u32, Ordering::Relaxed);
                parents[node.first_index as usize + 1].store(i as u32, Ordering::Relaxed);
            }
        });

        parents
            .into_iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    /// Basic debug print illustrating the bvh layout
    pub fn print_bvh(&self, node_index: usize, depth: usize) {
        let node = &self.nodes[node_index];
        if node.is_leaf() {
            println!(
                "{}{} leaf > {}",
                " ".repeat(depth),
                node_index,
                node.first_index
            )
        } else {
            println!(
                "{}{} inner > {}, {}",
                " ".repeat(depth),
                node_index,
                node.first_index,
                node.first_index + 1
            );
            self.print_bvh(node.first_index as usize, depth + 1);
            self.print_bvh(node.first_index as usize + 1, depth + 1);
        }
    }

    /// Get the maximum depth of the BVH from the given node
    pub fn depth(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        if node.is_leaf() {
            1
        } else {
            1 + self
                .depth(node.first_index as usize)
                .max(self.depth((node.first_index + 1) as usize))
        }
    }
}

/// Result of Bvh2 validation. Contains various bvh stats.
#[derive(Default)]
pub struct Bvh2ValidationResult {
    /// Whether the BVH primitives have splits or not.
    pub splits: bool,
    /// The primitives are already laid out in bvh.primitive_indices order.
    pub direct_layout: bool,
    /// Set of primitives discovered though validation traversal.
    pub discovered_primitives: HashSet<u32>,
    /// Set of nodes discovered though validation traversal.
    pub discovered_nodes: HashSet<u32>,
    /// Total number of nodes discovered though validation traversal.
    pub node_count: usize,
    /// Total number of leafs discovered though validation traversal.
    pub leaf_count: usize,
    /// Total number of primitives discovered though validation traversal.
    pub prim_count: usize,
    /// Maximum hierarchical BVH depth discovered though validation traversal.
    pub max_depth: u32,
    /// Quantity of nodes found at each depth though validation traversal.
    pub nodes_at_depth: HashMap<u32, u32>,
    /// Quantity of leaves found at each depth though validation traversal.
    pub leaves_at_depth: HashMap<u32, u32>,
}

impl fmt::Display for Bvh2ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GPU BVH Avg primitives/leaf: {:.3}",
            self.prim_count as f64 / self.leaf_count as f64
        )?;

        writeln!(
            f,
            "\
node_count: {}
prim_count: {}
leaf_count: {}",
            self.node_count, self.prim_count, self.leaf_count
        )?;

        writeln!(f, "Node & Leaf counts for each depth")?;
        for i in 0..=self.max_depth {
            writeln!(
                f,
                "{:<3} {:<10} {:<10}",
                i,
                self.nodes_at_depth.get(&i).unwrap_or(&0),
                self.leaves_at_depth.get(&i).unwrap_or(&0)
            )?;
        }

        Ok(())
    }
}
