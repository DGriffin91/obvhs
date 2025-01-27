//! A binary BVH

pub mod builder;
pub mod insertion_removal;
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
    Boundable, INVALID,
};

/// A node in the Bvh2, can be an inner node or leaf.
#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Bvh2Node {
    /// The bounding box for the primitive(s) contained in this node
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
#[derive(Clone, Default)]
pub struct Bvh2 {
    /// List of nodes contained in this bvh. first_index in Bvh2Node indexes into this list.
    pub nodes: Vec<Bvh2Node>,

    /// Mapping from bvh primitive indices to original input indices
    /// The reason for this mapping is that if multiple primitives are contained in a node, they need to have their indices laid out contiguously.
    /// To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub primitive_indices: Vec<u32>,

    /// A freelist for use when removing primitives from the bvh. These represent slots in Bvh2::primitive_indices
    /// that are available if a primitive is added to the bvh. Only currently used by Bvh2::remove_primitive() and
    /// Bvh2::insert_primitive() which are not part of the typical initial bvh generation.
    pub primitive_indices_freelist: Vec<u32>,

    /// An optional mapping from primitives back to nodes.
    /// Ex. let node_id = primitives_to_nodes.unwrap()[primitive_id];
    /// Where primitive_id is the original index of the primitive used when making the BVH and node_id is the index
    /// into Bvh2::nodes for the node of that primitive. Always use with the direct primitive id, not the one in the
    /// bvh node.
    /// If `primitives_to_nodes` is Some, it is expected that functions that modify the BVH will keep the mapping valid.
    pub primitives_to_nodes: Option<Vec<u32>>,

    /// An optional mapping from a given node index to that node's parent for each node in the bvh.
    /// If `parents` is Some, it is expected that functions that modify the BVH will keep the mapping valid.
    pub parents: Option<Vec<u32>>,

    /// This is set by operations that ensure that parents have higher indices than children and unset by operations
    /// that might disturb that order. Some operations require this ordering and will reorder if this is not true.
    pub children_are_ordered_after_parents: bool,

    /// Maximum bvh hierarchy depth. Used to determine stack depth for cpu bvh2 traversal.
    /// Stack defaults to 96 if max_depth isn't set, which much deeper than most bvh's even
    /// for large scenes without a tlas.
    pub max_depth: Option<usize>,
}
pub const DEFAULT_MAX_STACK_DEPTH: usize = 96;

impl Bvh2 {
    #[inline(always)]
    pub fn new_ray_traversal(&self, ray: Ray) -> RayTraversal {
        let mut stack =
            HeapStack::new_with_capacity(self.max_depth.unwrap_or(DEFAULT_MAX_STACK_DEPTH));
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
    /// Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    /// Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    /// original primitives per primitive_indices.
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
    /// Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    /// Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    /// original primitives per primitive_indices.
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
        let mut stack =
            HeapStack::new_with_capacity(self.max_depth.unwrap_or(DEFAULT_MAX_STACK_DEPTH));
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

    /// Order node array in stack traversal order. Ensures parents are always at lower indices than children. Fairly
    /// slow, can take around 1/3 of the time of building the same BVH from scratch from with the fastest_build preset.
    /// Doesn't seem to speed up traversal much for a new BVH created from PLOC, but if it has had many
    /// removals/insertions it can help.
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
        if self.parents.is_some() {
            self.update_parents();
        }
        if self.primitives_to_nodes.is_some() {
            self.update_primitives_to_nodes();
        }
        self.children_are_ordered_after_parents = true;
    }

    /// Refits the whole BVH from the leaves up. If the leaves have moved very much the BVH can quickly become
    /// degenerate causing significantly higher traversal times. Consider rebuilding the BVH from scratch or running a
    /// bit of reinsertion after refit.
    pub fn refit_all(&mut self) {
        if self.children_are_ordered_after_parents {
            // If children are already ordered after parents we can update in a single sweep.
            // Around 3x faster than the fallback below.
            for node_id in (0..self.nodes.len()).rev() {
                let node = &self.nodes[node_id];
                if !node.is_leaf() {
                    let first_child_bbox = self.nodes[node.first_index as usize].aabb;
                    let second_child_bbox = self.nodes[node.first_index as usize + 1].aabb;
                    self.nodes[node_id].aabb = first_child_bbox.union(&second_child_bbox);
                }
            }
        } else {
            // If not, we need to create a safe order in which we can make updates.
            // This is much faster than reordering the whole bvh with Bvh2::reorder_in_stack_traversal_order()
            let mut stack = HeapStack::new_with_capacity(self.max_depth.unwrap_or(1000));
            let mut reverse_stack = Vec::with_capacity(self.nodes.len());
            stack.push(0);
            reverse_stack.push(0);
            while let Some(current_node_index) = stack.pop() {
                let node = &self.nodes[*current_node_index as usize];
                if !node.is_leaf() {
                    reverse_stack.push(node.first_index);
                    reverse_stack.push(node.first_index + 1);
                    stack.push(node.first_index);
                    stack.push(node.first_index + 1);
                }
            }
            for node_id in reverse_stack.iter().rev() {
                let node = &self.nodes[*node_id as usize];
                if !node.is_leaf() {
                    let first_child_bbox = self.nodes[node.first_index as usize].aabb;
                    let second_child_bbox = self.nodes[node.first_index as usize + 1].aabb;
                    self.nodes[*node_id as usize].aabb = first_child_bbox.union(&second_child_bbox);
                }
            }
        }
    }

    /// Compute parents and update cache only if they have not already been computed
    pub fn init_parents(&mut self) {
        if self.parents.is_none() {
            self.update_parents();
        }
    }

    /// Compute the mapping from a given node index to that node's parent for each node in the bvh and update local
    /// cache.
    pub fn update_parents(&mut self) {
        self.parents = Some(self.compute_parents());
    }

    /// Compute the mapping from a given node index to that node's parent for each node in the bvh.
    pub fn compute_parents(&self) -> Vec<u32> {
        #[cfg(not(feature = "parallel"))]
        {
            let mut parents = vec![0; self.nodes.len()];
            parents[0] = 0;
            self.nodes.iter().enumerate().for_each(|(i, node)| {
                if !node.is_leaf() {
                    parents[node.first_index as usize] = i as u32;
                    parents[node.first_index as usize + 1] = i as u32;
                }
            });
            return parents;
        }
        // Seems around 80% faster than compute_parents.
        // TODO is there a better way to parallelize?
        #[cfg(feature = "parallel")]
        {
            let parents: Vec<AtomicU32> =
                (0..self.nodes.len()).map(|_| AtomicU32::new(0)).collect();

            parents[0].store(0, Ordering::Relaxed);

            self.nodes.par_iter().enumerate().for_each(|(i, node)| {
                if !node.is_leaf() {
                    parents[node.first_index as usize].store(i as u32, Ordering::Relaxed);
                    parents[node.first_index as usize + 1].store(i as u32, Ordering::Relaxed);
                }
            });

            return parents
                .into_iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
        }
    }

    /// Compute compute_primitives_to_nodes and update cache only if they have not already been computed
    pub fn init_primitives_to_nodes(&mut self) {
        if self.primitives_to_nodes.is_none() {
            self.update_primitives_to_nodes();
        }
    }

    /// Compute the mapping from primitive index to node index and update local cache.
    pub fn update_primitives_to_nodes(&mut self) {
        let mut primitives_to_nodes = vec![0u32; self.primitive_indices.len()];
        for (node_id, node) in self.nodes.iter().enumerate() {
            if node.is_leaf() {
                let start = node.first_index;
                let end = node.first_index + node.prim_count;
                for node_prim_id in start..end {
                    // TODO perf avoid this indirection by making self.primitive_indices optional?
                    let prim_id = self.primitive_indices[node_prim_id as usize];
                    primitives_to_nodes[prim_id as usize] = node_id as u32;
                }
            }
        }
        self.primitives_to_nodes = Some(primitives_to_nodes);
    }

    pub fn validate_parents(&self) {
        let parents = self.parents.as_deref().unwrap();
        self.nodes.iter().enumerate().for_each(|(i, node)| {
            if !node.is_leaf() {
                assert_eq!(parents[node.first_index as usize], i as u32);
                assert_eq!(parents[node.first_index as usize + 1], i as u32);
            }
        });
    }

    pub fn validate_primitives_to_nodes(&self) {
        let primitives_to_nodes = self.primitives_to_nodes.as_deref().unwrap();
        primitives_to_nodes
            .iter()
            .enumerate()
            .for_each(|(prim_id, node_id)| {
                if *node_id != INVALID {
                    let prim_id = prim_id as u32;
                    let node = &self.nodes[*node_id as usize];
                    assert!(node.is_leaf());
                    let start = node.first_index;
                    let end = node.first_index + node.prim_count;
                    let mut found = false;
                    for node_prim_id in start..end {
                        if prim_id == self.primitive_indices[node_prim_id as usize] {
                            found = true;
                            break;
                        }
                    }
                    assert!(found, "prim_id {prim_id} not found")
                }
            });
    }

    /// Refit the BVH working up the tree from this node, ignoring leaves. (TODO add a version that checks leaves)
    /// This recomputes the Aabbs for all the parents of the given node index.
    /// This can only be used to refit when a single node has changed or moved.
    pub fn refit_from(&mut self, mut index: usize) {
        self.init_parents();
        let parents = self.parents.as_deref().unwrap();
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
    pub fn refit_from_fast(&mut self, mut index: usize) {
        self.init_parents();
        let parents = self.parents.as_deref().unwrap();
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

    /// Get the count of active primitive indices.
    /// when primitives are removed they are added to the `primitive_indices_freelist` so the
    /// self.primitive_indices.len() may not represent the actual number of valid, active primitive_indices.
    #[inline(always)]
    pub fn active_primitive_indices_count(&self) -> usize {
        self.primitive_indices.len() - self.primitive_indices_freelist.len()
    }

    /// Direct layout: The primitives are already laid out in bvh.primitive_indices order.
    pub fn validate<T: Boundable>(
        &self,
        primitives: &[T],
        direct_layout: bool,
        splits: bool,
        tight_fit: bool,
    ) -> Bvh2ValidationResult {
        if self.primitives_to_nodes.is_some() {
            self.validate_primitives_to_nodes();
        }

        if self.parents.is_some() {
            self.validate_parents();
        }

        let mut result = Bvh2ValidationResult {
            splits,
            direct_layout,
            require_tight_fit: tight_fit,
            ..Default::default()
        };

        if !self.nodes.is_empty() {
            self.validate_impl::<T>(primitives, &mut result, 0, 0, 0);
        }
        assert_eq!(result.discovered_nodes.len(), self.nodes.len());
        assert_eq!(result.node_count, self.nodes.len());
        if !direct_layout {
            // Ignore primitive_indices if this is a direct layout
            let active_indices_count = self.active_primitive_indices_count();
            assert_eq!(result.discovered_primitives.len(), active_indices_count);
            assert_eq!(result.prim_count, active_indices_count);
        }
        assert!(result.max_depth < self.max_depth.unwrap_or(DEFAULT_MAX_STACK_DEPTH) as u32);

        if self.children_are_ordered_after_parents {
            // Assert that children are always ordered after parents in self.nodes
            let temp_parents;
            let parents = if let Some(parents) = self.parents.as_ref() {
                parents
            } else {
                temp_parents = self.compute_parents();
                &temp_parents
            };
            for node_id in (1..self.nodes.len()).rev() {
                assert!(parents[node_id] < node_id as u32);
            }
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
            let mut temp_aabb = Aabb::empty();
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
                    temp_aabb = temp_aabb.union(&prim_aabb);
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
            if result.require_tight_fit {
                assert_eq!(
                    temp_aabb, node.aabb,
                    "Primitive do not fit in tightly in parent {node_index}",
                );
            }
        } else {
            if result.require_tight_fit {
                let left_id = node.first_index as usize;
                let right_id = node.first_index as usize + 1;
                let left_child_aabb = &self.nodes[left_id];
                let right_child_aabb = &self.nodes[right_id];

                assert_eq!(
                    left_child_aabb.aabb.union(&right_child_aabb.aabb),
                    node.aabb,
                    "Children {left_id} & {right_id} do not fit in tightly in parent {node_index}",
                );
            }

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

/// Update the `primitives_to_nodes` mappings for primitives contained in `node_id`. Does nothing if primitives_to_nodes
/// is not already init.
// Not a member of Bvh2 because of borrow issues when a reference to other things like parents is also taken.
// Maybe could be cleaner as a macro?
#[inline]
fn update_primitives_to_nodes_for_node(
    node: &Bvh2Node,
    node_id: usize,
    primitive_indices: &[u32],
    primitives_to_nodes: &mut Option<Vec<u32>>,
) {
    if let Some(primitives_to_nodes) = primitives_to_nodes.as_mut() {
        let start = node.first_index;
        let end = start + node.prim_count;
        for node_prim_id in start..end {
            let direct_prim_id = primitive_indices[node_prim_id as usize];
            primitives_to_nodes[direct_prim_id as usize] = node_id as u32;
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
    /// Require validation to ensure aabbs tightly fit children and primitives.
    pub require_tight_fit: bool,
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

#[cfg(test)]
mod tests {

    use glam::*;

    use crate::{
        ploc::{PlocSearchDistance, SortPrecision},
        test_util::geometry::demoscene,
        Transformable,
    };

    #[test]
    fn test_refit_all() {
        let mut tris = demoscene(32, 0);
        let mut aabbs = Vec::with_capacity(tris.len());
        let mut indices = Vec::with_capacity(tris.len());
        for (i, primitive) in tris.iter().enumerate() {
            indices.push(i as u32);
            aabbs.push(primitive.aabb());
        }

        // Test without init_primitives_to_nodes & init_parents
        let mut bvh =
            PlocSearchDistance::VeryLow.build(&aabbs, indices.clone(), SortPrecision::U64, 1);

        bvh.init_primitives_to_nodes();
        let primitives_to_nodes = bvh.primitives_to_nodes.as_ref().unwrap();
        tris.transform(&Mat4::from_scale_rotation_translation(
            Vec3::splat(1.3),
            Quat::from_rotation_y(0.1),
            vec3(0.33, 0.3, 0.37),
        ));
        for (prim_id, tri) in tris.iter().enumerate() {
            bvh.nodes[primitives_to_nodes[prim_id] as usize].aabb = tri.aabb();
        }

        bvh.refit_all();

        bvh.validate(&tris, false, false, true);
    }
}
