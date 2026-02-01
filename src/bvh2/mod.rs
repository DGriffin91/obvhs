//! A binary BVH

pub mod builder;
pub mod insertion_removal;
pub mod leaf_collapser;

pub mod node;
pub mod reinsertion;

use bytemuck::zeroed_vec;
use glam::Vec3A;
use node::Bvh2Node;

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use reinsertion::find_reinsertion;

use crate::{
    Boundable, INVALID,
    aabb::Aabb,
    fast_stack,
    faststack::FastStack,
    ray::{Ray, RayHit},
};

/// A binary BVH
#[derive(Clone)]
pub struct Bvh2 {
    /// List of nodes contained in this bvh. first_index in Bvh2Node for inner nodes indexes into this list. This list
    /// fully represents the BVH tree. The other fields in this struct provide additional information that allow the BVH
    /// to be manipulated more efficiently, but are not actually part of the BVH itself. The only other critical field is
    /// `primitive_indices`, assuming the BVH is not using a direct mapping.
    pub nodes: Vec<Bvh2Node>,

    /// Mapping from bvh primitive indices to original input indices
    /// The reason for this mapping is that if multiple primitives are contained in a node, they need to have their
    /// indices laid out contiguously. To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the
    ///    primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub primitive_indices: Vec<u32>,

    /// A freelist for use when removing primitives from the bvh. These represent slots in Bvh2::primitive_indices
    /// that are available if a primitive is added to the bvh. Only currently used by Bvh2::remove_primitive() and
    /// Bvh2::insert_primitive() which are not part of the typical initial bvh generation.
    pub primitive_indices_freelist: Vec<u32>,

    /// An optional mapping from primitives back to nodes.
    /// Ex. `let node_id = primitives_to_nodes[primitive_id];`
    /// Where primitive_id is the original index of the primitive used when making the BVH and node_id is the index
    /// into Bvh2::nodes for the node of that primitive. Always use with the direct primitive id, not the one in the
    /// bvh node.
    /// See: Bvh2::init_primitives_to_nodes().
    /// If `primitives_to_nodes` is empty it's expected that it has not been initialized yet or has been invalidated.
    /// If `primitives_to_nodes` is not empty, it is expected that functions that modify the BVH will keep the mapping
    /// valid.
    pub primitives_to_nodes: Vec<u32>,

    /// An optional mapping from a given node index to that node's parent for each node in the bvh.
    /// See: Bvh2::init_parents_if_uninit().
    /// If `parents` is empty it's expected that it has not been initialized yet or has been invalidated.
    /// If `parents` is not empty it's expected that functions that modify the BVH will keep the mapping valid.
    pub parents: Vec<u32>,

    /// This is set by operations that ensure that parents have higher indices than children and unset by operations
    /// that might disturb that order. Some operations require this ordering and will reorder if this is not true.
    pub children_are_ordered_after_parents: bool,

    /// Stack defaults to 96 or the max depth found during initial ploc building, whichever is larger. This may be
    /// larger than needed depending on what post processing steps (like collapse, reinsertion, etc...), but the cost of
    /// recalculating it may not be worth it so it is not done automatically.
    pub max_depth: usize,

    /// Indicates that this BVH is using spatial splits. Large triangles are split into multiple smaller Aabbs, so
    /// primitives will extend outside the leaf in some cases.
    /// If the bvh uses splits, a primitive can show up in multiple leaf nodes so there wont be a 1 to 1 correlation
    /// between the total number of primitives in leaf nodes and in Bvh2::primitive_indices, vs the input triangles.
    /// If spatial splits are used, some validation steps have to be skipped and some features are unavailable:
    /// `Bvh2::add_leaf()`, `Bvh2::remove_leaf()`, `Bvh2::add_primitive()`, `Bvh2::remove_primitive()` as these would
    /// require a mapping from one primitive to multiple nodes in `Bvh2::primitives_to_nodes`
    pub uses_spatial_splits: bool,
}

pub const DEFAULT_MAX_STACK_DEPTH: usize = 96;

impl Default for Bvh2 {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            primitive_indices: Default::default(),
            primitive_indices_freelist: Default::default(),
            primitives_to_nodes: Default::default(),
            parents: Default::default(),
            children_are_ordered_after_parents: Default::default(),
            max_depth: DEFAULT_MAX_STACK_DEPTH,
            uses_spatial_splits: Default::default(),
        }
    }
}

impl Bvh2 {
    /// Reset BVH while keeping allocations for rebuild. Note: results in an invalid bvh until rebuilt.
    pub fn reset_for_reuse(&mut self, prim_count: usize, indices: Option<Vec<u32>>) {
        let nodes_count = (2 * prim_count as i64 - 1).max(0) as usize;
        self.nodes.resize(nodes_count, Default::default());
        if let Some(indices) = indices {
            self.primitive_indices = indices;
        } else {
            self.primitive_indices
                .resize(prim_count, Default::default());
        }
        self.primitive_indices_freelist.clear();
        self.primitives_to_nodes.clear();
        self.parents.clear();
        self.children_are_ordered_after_parents = Default::default();
        self.max_depth = DEFAULT_MAX_STACK_DEPTH;
        self.uses_spatial_splits = Default::default();
    }

    pub fn zeroed(prim_count: usize) -> Self {
        let nodes_count = (2 * prim_count as i64 - 1).max(0) as usize;
        Self {
            nodes: zeroed_vec(nodes_count),
            primitive_indices: zeroed_vec(prim_count),
            primitive_indices_freelist: Default::default(),
            primitives_to_nodes: Default::default(),
            parents: Default::default(),
            children_are_ordered_after_parents: Default::default(),
            max_depth: DEFAULT_MAX_STACK_DEPTH,
            uses_spatial_splits: Default::default(),
        }
    }

    /// Traverse the bvh for a given `Ray`. Returns the closest intersected primitive.
    ///
    /// # Arguments
    /// * `ray` - The ray to be tested for intersection.
    /// * `hit` - As traverse_dynamic intersects primitives, it will update `hit` with the closest.
    /// * `intersection_fn` - should take the given ray and primitive index and return the distance to the intersection, if any.
    ///
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
        let mut intersect_prims = |node: &Bvh2Node, ray: &mut Ray, hit: &mut RayHit| {
            (node.first_index..node.first_index + node.prim_count).for_each(|primitive_id| {
                let t = intersection_fn(ray, primitive_id as usize);
                if t < ray.tmax {
                    hit.primitive_id = primitive_id;
                    hit.t = t;
                    ray.tmax = t;
                }
            });
            true
        };

        fast_stack!(u32, (96, 192), self.max_depth, stack, {
            Bvh2::ray_traverse_dynamic(self, &mut stack, ray, hit, &mut intersect_prims)
        });

        hit.t < ray.tmax // Note this is valid since traverse_with_stack does not mutate the ray
    }

    /// Traverse the bvh for a given `Ray`. Returns true if the ray missed all primitives.
    ///
    /// # Arguments
    /// * `ray` - The ray to be tested for intersection.
    /// * `hit` - As traverse_dynamic intersects primitives, it will update `hit` with the closest.
    /// * `intersection_fn` - should take the given ray and primitive index and return the distance to the intersection, if any.
    ///
    /// Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    /// Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    /// original primitives per primitive_indices.
    #[inline(always)]
    pub fn ray_traverse_miss<F: FnMut(&Ray, usize) -> f32>(
        &self,
        ray: Ray,
        mut intersection_fn: F,
    ) -> bool {
        let mut miss = true;
        let mut intersect_prims = |node: &Bvh2Node, ray: &mut Ray, _hit: &mut RayHit| {
            for primitive_id in node.first_index..node.first_index + node.prim_count {
                let t = intersection_fn(ray, primitive_id as usize);
                if t < ray.tmax {
                    miss = false;
                    return false;
                }
            }
            true
        };

        fast_stack!(u32, (96, 192), self.max_depth, stack, {
            Bvh2::ray_traverse_dynamic(
                self,
                &mut stack,
                ray,
                &mut RayHit::none(),
                &mut intersect_prims,
            )
        });

        miss
    }

    /// Traverse the bvh for a given `Ray`. Intersects all primitives along ray (for things like evaluating transparency)
    ///   intersection_fn is called for all intersections. Ray is not updated to allow for evaluating at every hit.
    ///
    /// # Arguments
    /// * `ray` - The ray to be tested for intersection.
    /// * `intersection_fn` - takes the given ray and primitive index.
    ///
    /// Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    /// Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    /// original primitives per primitive_indices.
    #[inline(always)]
    pub fn ray_traverse_anyhit<F: FnMut(&Ray, usize)>(&self, ray: Ray, mut intersection_fn: F) {
        let mut intersect_prims = |node: &Bvh2Node, ray: &mut Ray, _hit: &mut RayHit| {
            for primitive_id in node.first_index..node.first_index + node.prim_count {
                intersection_fn(ray, primitive_id as usize);
            }
            true
        };

        let mut hit = RayHit::none();
        fast_stack!(u32, (96, 192), self.max_depth, stack, {
            self.ray_traverse_dynamic(&mut stack, ray, &mut hit, &mut intersect_prims)
        });
    }

    /// Traverse the BVH
    /// Returns false when no hit is found. Consider using or referencing: Bvh2::ray_traverse(),
    /// Bvh2::ray_traverse_miss(), or Bvh2::ray_traverse_anyhit().
    ///
    /// # Arguments
    /// * `state` - Holds the current traversal state. Allows traverse_dynamic to yield.
    /// * `hit` - As traverse_dynamic intersects primitives, it will update `hit` with the closest.
    /// * `intersection_fn` - should test the primitives in the given node, update the ray.tmax, and hit info. Return
    ///   false to halt traversal.
    ///   For basic miss test return false on first hit to halt traversal.
    ///   For closest hit run until it returns false and check hit.t < ray.tmax to see if it hit something
    ///   For transparency, you want to hit every primitive in the ray's path, keeping track of the closest opaque hit.
    ///   and then manually setting ray.tmax to that closest opaque hit at each iteration.
    ///
    /// Note the primitive index should index first into Bvh2::primitive_indices then that will be index of original primitive.
    /// Various parts of the BVH building process might reorder the primitives. To avoid this indirection, reorder your
    /// original primitives per primitive_indices.
    #[inline(always)]
    pub fn ray_traverse_dynamic<
        F: FnMut(&Bvh2Node, &mut Ray, &mut RayHit) -> bool,
        Stack: FastStack<u32>,
    >(
        &self,
        stack: &mut Stack,
        mut ray: Ray,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let root_node = &self.nodes[0];
        let hit_root = root_node.aabb().intersect_ray(&ray) < ray.tmax;
        if !hit_root {
            return;
        } else if root_node.is_leaf() {
            intersection_fn(root_node, &mut ray, hit);
            return;
        };

        let mut current_node_index = root_node.first_index;
        loop {
            let right_index = current_node_index as usize + 1;
            assert!(right_index < self.nodes.len());
            let mut left_node = unsafe { self.nodes.get_unchecked(current_node_index as usize) };
            let mut right_node = unsafe { self.nodes.get_unchecked(right_index) };

            // TODO perf: could it be faster to intersect these at the same time with avx?
            let mut left_t = left_node.aabb().intersect_ray(&ray);
            let mut right_t = right_node.aabb().intersect_ray(&ray);

            if left_t > right_t {
                core::mem::swap(&mut left_t, &mut right_t);
                core::mem::swap(&mut left_node, &mut right_node);
            }

            let hit_left = left_t < ray.tmax;

            let go_left = if hit_left && left_node.is_leaf() {
                if !intersection_fn(left_node, &mut ray, hit) {
                    return;
                }
                false
            } else {
                hit_left
            };

            let hit_right = right_t < ray.tmax;

            let go_right = if hit_right && right_node.is_leaf() {
                if !intersection_fn(right_node, &mut ray, hit) {
                    return;
                }
                false
            } else {
                hit_right
            };

            match (go_left, go_right) {
                (true, true) => {
                    current_node_index = left_node.first_index;
                    stack.push(right_node.first_index);
                }
                (true, false) => current_node_index = left_node.first_index,
                (false, true) => current_node_index = right_node.first_index,
                (false, false) => {
                    let Some(next) = stack.pop() else {
                        hit.t = ray.tmax;
                        return;
                    };
                    current_node_index = next;
                }
            }
        }
    }

    /// Recursively traverse the bvh for a given `Ray`.
    /// On completion, `leaf_indices` will contain a list of the intersected leaf node indices.
    /// This method is slower than stack traversal and only exists as a reference.
    /// This method does not check if the primitive was intersected, only the leaf node.
    pub fn ray_traverse_recursive(
        &self,
        ray: &Ray,
        node_index: usize,
        leaf_indices: &mut Vec<usize>,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let node = &self.nodes[node_index];
        if node.aabb().intersect_ray(ray) < f32::INFINITY {
            if node.is_leaf() {
                leaf_indices.push(node_index);
            } else {
                self.ray_traverse_recursive(ray, node.first_index as usize, leaf_indices);
                self.ray_traverse_recursive(ray, node.first_index as usize + 1, leaf_indices);
            }
        }
    }

    /// Traverse the BVH with an Aabb. fn `eval` is called for nodes that intersect `aabb`
    /// The bvh (self) and the current node index is passed into fn `eval`
    /// Note each node may have multiple primitives. `node.first_index` is the index of the first primitive.
    /// `node.prim_count` is the quantity of primitives contained in the given node.
    /// Return false from eval to halt traversal
    pub fn aabb_traverse<F: FnMut(&Self, u32) -> bool>(&self, aabb: Aabb, mut eval: F) {
        if self.nodes.is_empty() {
            return;
        }

        let root_node = &self.nodes[0];
        if root_node.is_leaf() {
            if root_node.aabb().intersect_aabb(&aabb) {
                eval(self, 0);
            }
            return;
        }

        fast_stack!(u32, (96, 192), self.max_depth, stack, {
            stack.push(root_node.first_index);
            while let Some(node_index) = stack.pop() {
                // Left
                let node = &self.nodes[node_index as usize];
                if node.aabb().intersect_aabb(&aabb) {
                    if node.is_leaf() {
                        if !eval(self, node_index) {
                            return;
                        }
                    } else {
                        stack.push(node.first_index);
                    }
                }

                // Right
                let node_index = node_index + 1;
                let node = &self.nodes[node_index as usize];
                if node.aabb().intersect_aabb(&aabb) {
                    if node.is_leaf() {
                        if !eval(self, node_index) {
                            return;
                        }
                    } else {
                        stack.push(node.first_index);
                    }
                }
            }
        });
    }

    /// Traverse the BVH with a point. fn `eval` is called for nodes that intersect `point`
    /// The bvh (self) and the current node index is passed into fn `eval`
    /// Note each node may have multiple primitives. `node.first_index` is the index of the first primitive.
    /// `node.prim_count` is the quantity of primitives contained in the given node.
    /// Return false from eval to halt traversal
    pub fn point_traverse<F: FnMut(&Self, u32) -> bool>(&self, point: Vec3A, mut eval: F) {
        if self.nodes.is_empty() {
            return;
        }

        let root_node = &self.nodes[0];
        if root_node.is_leaf() {
            if root_node.aabb().contains_point(point) {
                eval(self, 0);
            }
            return;
        }

        fast_stack!(u32, (96, 192), self.max_depth, stack, {
            stack.push(root_node.first_index);
            while let Some(node_index) = stack.pop() {
                // Left
                let node = &self.nodes[node_index as usize];
                if node.aabb().contains_point(point) {
                    if node.is_leaf() {
                        if !eval(self, node_index) {
                            return;
                        }
                    } else {
                        stack.push(node.first_index);
                    }
                }

                // Right
                let node_index = node_index + 1;
                let node = &self.nodes[node_index as usize];
                if node.aabb().contains_point(point) {
                    if node.is_leaf() {
                        if !eval(self, node_index) {
                            return;
                        }
                    } else {
                        stack.push(node.first_index);
                    }
                }
            }
        });
    }

    /// Order node array in stack traversal order. Ensures parents are always at lower indices than children. Fairly
    /// slow, can take around 1/3 of the time of building the same BVH from scratch from with the fastest_build preset.
    /// Doesn't seem to speed up traversal much for a new BVH created from PLOC, but if it has had many
    /// removals/insertions it can help.
    pub fn reorder_in_stack_traversal_order(&mut self) {
        if self.nodes.len() < 2 {
            return;
        }
        let mut new_nodes: Vec<Bvh2Node> = Vec::with_capacity(self.nodes.len());
        let mut mapping = vec![0; self.nodes.len()]; // Map from where n node used to be to where it is now
        let mut stack = Vec::new();
        stack.push(self.nodes[0].first_index);
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
        if !self.parents.is_empty() {
            self.update_parents();
        }
        if !self.primitives_to_nodes.is_empty() {
            self.update_primitives_to_nodes();
        }
        self.children_are_ordered_after_parents = true;
    }

    /// Refits the whole BVH from the leaves up. If the leaves have moved very much the BVH can quickly become
    /// degenerate causing significantly higher traversal times. Consider rebuilding the BVH from scratch or running a
    /// bit of reinsertion after refit.
    /// Usage:
    /// ```
    ///    use glam::*;
    ///    use obvhs::*;
    ///    use obvhs::{ploc::*, test_util::geometry::demoscene, bvh2::builder::build_bvh2_from_tris};
    ///    use std::time::Duration;
    ///
    ///    let mut tris = demoscene(32, 0);
    ///    let mut bvh = build_bvh2_from_tris(&tris, BvhBuildParams::fastest_build(), &mut Duration::default());
    ///
    ///    bvh.init_primitives_to_nodes_if_uninit(); // Generate mapping from primitives to nodes
    ///    tris.transform(&Mat4::from_scale_rotation_translation(
    ///        Vec3::splat(1.3),
    ///        Quat::from_rotation_y(0.1),
    ///        vec3(0.33, 0.3, 0.37),
    ///    ));
    ///    for (prim_id, tri) in tris.iter().enumerate() {
    ///        bvh.nodes[bvh.primitives_to_nodes[prim_id] as usize].set_aabb(tri.aabb()); // Update aabbs
    ///    }
    ///    bvh.refit_all(); // Refit aabbs
    ///    bvh.validate(&tris, false, true); // Validate that aabbs are now fitting tightly
    /// ```
    pub fn refit_all(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        if self.children_are_ordered_after_parents {
            // If children are already ordered after parents we can update in a single sweep.
            // Around 3x faster than the fallback below.
            for node_id in (0..self.nodes.len()).rev() {
                let node = &self.nodes[node_id];
                if !node.is_leaf() {
                    let first_child_bbox = *self.nodes[node.first_index as usize].aabb();
                    let second_child_bbox = *self.nodes[node.first_index as usize + 1].aabb();
                    self.nodes[node_id].set_aabb(first_child_bbox.union(&second_child_bbox));
                }
            }
        } else {
            // If not, we need to create a safe order in which we can make updates.
            // This is much faster than reordering the whole bvh with Bvh2::reorder_in_stack_traversal_order()
            fast_stack!(u32, (96, 192), self.max_depth, stack, {
                let mut reverse_stack = Vec::with_capacity(self.nodes.len());
                stack.push(0);
                reverse_stack.push(0);
                while let Some(current_node_index) = stack.pop() {
                    let node = &self.nodes[current_node_index as usize];
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
                        let first_child_bbox = *self.nodes[node.first_index as usize].aabb();
                        let second_child_bbox = *self.nodes[node.first_index as usize + 1].aabb();
                        self.nodes[*node_id as usize]
                            .set_aabb(first_child_bbox.union(&second_child_bbox));
                    }
                }
            });
        }
    }

    /// Compute parents and update cache only if they have not already been computed
    pub fn init_parents_if_uninit(&mut self) {
        if self.parents.is_empty() {
            self.update_parents();
        }
    }

    /// Compute the mapping from a given node index to that node's parent for each node in the bvh and update local
    /// cache.
    pub fn update_parents(&mut self) {
        Bvh2::compute_parents(&self.nodes, &mut self.parents);
    }

    /// Compute the mapping from a given node index to that node's parent for each node in the bvh, takes a Vec to allow
    /// reusing the allocation.
    pub fn compute_parents(nodes: &[Bvh2Node], parents: &mut Vec<u32>) {
        parents.resize(nodes.len(), 0);

        if nodes.is_empty() {
            return;
        }

        parents[0] = 0;

        #[cfg(not(feature = "parallel"))]
        {
            nodes.iter().enumerate().for_each(|(i, node)| {
                if !node.is_leaf() {
                    parents[node.first_index as usize] = i as u32;
                    parents[node.first_index as usize + 1] = i as u32;
                }
            });
        }
        // Seems around 80% faster than compute_parents.
        // TODO is there a better way to parallelize?
        #[cfg(feature = "parallel")]
        {
            use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
            use std::sync::atomic::Ordering;

            let parents = crate::as_slice_of_atomic_u32(parents);
            nodes.par_iter().enumerate().for_each(|(i, node)| {
                if !node.is_leaf() {
                    parents[node.first_index as usize].store(i as u32, Ordering::Relaxed);
                    parents[node.first_index as usize + 1].store(i as u32, Ordering::Relaxed);
                }
            });
        }
    }

    /// Compute compute_primitives_to_nodes and update cache only if they have not already been computed. Not supported
    /// if using spatial splits as it would require a mapping from one primitive to multiple nodes.
    pub fn init_primitives_to_nodes_if_uninit(&mut self) {
        if self.primitives_to_nodes.is_empty() {
            self.update_primitives_to_nodes();
        }
    }

    /// Compute the mapping from primitive index to node index and update local cache. Not supported if using spatial
    /// splits as it would require a mapping from one primitive to multiple nodes.
    pub fn update_primitives_to_nodes(&mut self) {
        if self.uses_spatial_splits {
            log::warn!(
                "Calculating primitives_to_nodes while using spatial splits is currently unsupported as it would \
                require a mapping from one primitive to multiple nodes in `Bvh2::primitives_to_nodes`."
            );
        }

        Bvh2::compute_primitives_to_nodes(
            &self.nodes,
            &self.primitive_indices,
            &mut self.primitives_to_nodes,
        );
    }

    /// Compute the mapping from primitive index to node index. Takes a Vec to allow reusing the allocation.
    pub fn compute_primitives_to_nodes(
        nodes: &[Bvh2Node],
        primitive_indices: &[u32],
        primitives_to_nodes: &mut Vec<u32>,
    ) {
        primitives_to_nodes.clear();
        primitives_to_nodes.resize(primitive_indices.len(), INVALID);
        for (node_id, node) in nodes.iter().enumerate() {
            if node.is_leaf() {
                let start = node.first_index;
                let end = node.first_index + node.prim_count;
                for node_prim_id in start..end {
                    // TODO perf avoid this indirection by making self.primitive_indices optional?
                    let prim_id = primitive_indices[node_prim_id as usize];
                    primitives_to_nodes[prim_id as usize] = node_id as u32;
                }
            }
        }
    }

    pub fn validate_parents(&self) {
        self.nodes.iter().enumerate().for_each(|(i, node)| {
            if !node.is_leaf() {
                assert_eq!(self.parents[node.first_index as usize], i as u32);
                assert_eq!(self.parents[node.first_index as usize + 1], i as u32);
            }
        });
    }

    pub fn validate_primitives_to_nodes(&self) {
        self.primitives_to_nodes
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
        self.init_parents_if_uninit();
        loop {
            let node = &self.nodes[index];
            if !node.is_leaf() {
                let first_child_bbox = *self.nodes[node.first_index as usize].aabb();
                let second_child_bbox = *self.nodes[node.first_index as usize + 1].aabb();
                self.nodes[index].set_aabb(first_child_bbox.union(&second_child_bbox));
            }
            if index == 0 {
                break;
            }
            index = self.parents[index] as usize;
        }
    }

    /// Refit the BVH working up the tree from this node, ignoring leaves.
    /// This recomputes the Aabbs for the parents of the given node index.
    /// Halts if the parents are the same size. Panics in debug if some parents still needed to be resized.
    /// This can only be used to refit when a single node has changed or moved.
    pub fn refit_from_fast(&mut self, mut index: usize) {
        self.init_parents_if_uninit();
        let mut same_count = 0;
        loop {
            let node = &self.nodes[index];
            if !node.is_leaf() {
                let first_child_bbox = self.nodes[node.first_index as usize].aabb();
                let second_child_bbox = self.nodes[node.first_index as usize + 1].aabb();
                let new_aabb = first_child_bbox.union(second_child_bbox);
                let node = &mut self.nodes[index];
                if node.aabb() == &new_aabb {
                    same_count += 1;
                    #[cfg(not(debug_assertions))]
                    if same_count == 2 {
                        return;
                    }
                } else {
                    debug_assert!(
                        same_count < 2,
                        "Some parents still needed refitting. Unideal fitting is occurring somewhere."
                    );
                }
                node.set_aabb(new_aabb);
            }
            if index == 0 {
                break;
            }
            index = self.parents[index] as usize;
        }
    }

    /// Update node aabb and refit the BVH working up the tree from this node.
    #[inline]
    pub fn resize_node(&mut self, node_id: usize, aabb: Aabb) {
        self.nodes[node_id].set_aabb(aabb);
        self.refit_from_fast(node_id);
    }

    /// Find if there might be a better spot in the BVH for this node and move it there. The id of the reinserted node
    /// does not changed.
    #[inline]
    pub fn reinsert_node(&mut self, node_id: usize) {
        if node_id == 0 {
            return;
        }
        let reinsertion = find_reinsertion(self, node_id);
        if reinsertion.area_diff > 0.0 {
            reinsertion::reinsert_node(self, reinsertion.from as usize, reinsertion.to as usize);
            self.children_are_ordered_after_parents = false;
        }
    }

    /// Get the count of active primitive indices.
    /// when primitives are removed they are added to the `primitive_indices_freelist` so the
    /// self.primitive_indices.len() may not represent the actual number of valid, active primitive_indices.
    #[inline(always)]
    pub fn active_primitive_indices_count(&self) -> usize {
        self.primitive_indices.len() - self.primitive_indices_freelist.len()
    }

    /// direct_layout: The primitives are already laid out in bvh.primitive_indices order.
    /// tight_fit: Requires that children nodes and primitives fit tightly in parents. This is ignored for primitives
    ///     if the bvh uses spatial splits (tight_fit can still be set to `true`). This was added for validating
    ///     refit_all().
    pub fn validate<T: Boundable>(
        &self,
        primitives: &[T],
        direct_layout: bool,
        tight_fit: bool,
    ) -> Bvh2ValidationResult {
        let mut result = Bvh2ValidationResult {
            direct_layout,
            require_tight_fit: tight_fit,
            ..Default::default()
        };

        if self.nodes.is_empty() {
            assert!(self.parents.is_empty());
            assert!(self.primitives_to_nodes.is_empty());
            return result;
        }

        if !self.primitives_to_nodes.is_empty() {
            self.validate_primitives_to_nodes();
        }

        if !self.parents.is_empty() {
            self.validate_parents();
        }

        if !self.nodes.is_empty() {
            self.validate_impl::<T>(primitives, &mut result, 0, 0, 0);
        }
        assert_eq!(result.discovered_nodes.len(), self.nodes.len());
        assert_eq!(result.node_count, self.nodes.len());

        // Ignore primitive_indices if this is a direct layout
        if !direct_layout {
            if result.discovered_primitives.is_empty() {
                assert!(self.active_primitive_indices_count() == 0)
            } else {
                if !self.uses_spatial_splits {
                    // If the bvh uses splits, a primitive can show up in multiple leaf nodes so there wont be a 1 to 1
                    // correlation between the number of discovered primitives and the quantity in bvh.primitive_indices.
                    let active_indices_count = self.active_primitive_indices_count();
                    assert_eq!(result.discovered_primitives.len(), active_indices_count);
                    assert_eq!(result.prim_count, active_indices_count);
                }
                // Check that the set of discovered_primitives is the same as the set in primitive_indices while
                // ignoring empty slots in primitive_indices.
                let primitive_indices_freeset: HashSet<&u32> =
                    HashSet::from_iter(&self.primitive_indices_freelist);
                for (slot, index) in self.primitive_indices.iter().enumerate() {
                    let slot = slot as u32;
                    if !primitive_indices_freeset.contains(&slot) {
                        assert!(result.discovered_primitives.contains(index));
                    }
                }
                let primitive_indices_set: HashSet<&u32> =
                    HashSet::from_iter(self.primitive_indices.iter().filter(|i| **i != INVALID));
                for discovered_prim_id in &result.discovered_primitives {
                    assert!(primitive_indices_set.contains(discovered_prim_id))
                }
            }
        }
        assert!(
            result.max_depth < self.max_depth as u32,
            "result.max_depth ({}) must be less than self.max_depth ({})",
            result.max_depth,
            self.max_depth as u32
        );
        if result.max_depth > DEFAULT_MAX_STACK_DEPTH as u32 {
            log::warn!(
                "bvh depth is: {}, a depth beyond {} may be indicative of something pathological in the scene (like thousands of instances perfectly overlapping geometry) that will result in a BVH that is very slow to traverse.",
                result.max_depth,
                DEFAULT_MAX_STACK_DEPTH
            );
        }

        if self.children_are_ordered_after_parents {
            // Assert that children are always ordered after parents in self.nodes
            let mut temp_parents = vec![];
            let parents = if self.parents.is_empty() {
                Bvh2::compute_parents(&self.nodes, &mut temp_parents);
                &temp_parents
            } else {
                &self.parents
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
        let parent_aabb = self.nodes[parent_index as usize].aabb();
        result.discovered_nodes.insert(node_index);
        let node = &self.nodes[node_index as usize];
        result.node_count += 1;

        if let Some(count) = result.nodes_at_depth.get(&current_depth) {
            result.nodes_at_depth.insert(current_depth, count + 1);
        } else {
            result.nodes_at_depth.insert(current_depth, 1);
        }

        assert!(
            node.aabb().min.cmpge(parent_aabb.min).all()
                && node.aabb().max.cmple(parent_aabb.max).all(),
            "Child {} does not fit in parent {}:\nchild:  {:?}\nparent: {:?}",
            node_index,
            parent_index,
            node.aabb(),
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
                if result.direct_layout {
                    result.discovered_primitives.insert(prim_index as u32);
                } else {
                    result
                        .discovered_primitives
                        .insert(self.primitive_indices[prim_index]);
                }
                // If using splits, primitives will extend outside the leaf in some cases.
                if !self.uses_spatial_splits {
                    if !result.direct_layout {
                        prim_index = self.primitive_indices[prim_index] as usize
                    }
                    let prim_aabb = primitives[prim_index].aabb();
                    temp_aabb = temp_aabb.union(&prim_aabb);
                    assert!(
                        prim_aabb.min.cmpge(node.aabb().min).all()
                            && prim_aabb.max.cmple(node.aabb().max).all(),
                        "Primitive {} does not fit in parent {}:\nprimitive: {:?}\nparent:    {:?}",
                        prim_index,
                        parent_index,
                        prim_aabb,
                        node.aabb()
                    );
                }
            }
            if result.require_tight_fit && !self.uses_spatial_splits {
                assert_eq!(
                    temp_aabb,
                    *node.aabb(),
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
                    left_child_aabb.aabb().union(right_child_aabb.aabb()),
                    *node.aabb(),
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
    primitives_to_nodes: &mut [u32],
) {
    if !primitives_to_nodes.is_empty() {
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
    /// Total number of leaves discovered though validation traversal.
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
        BvhBuildParams, Transformable,
        ploc::{PlocBuilder, PlocSearchDistance, SortPrecision},
        test_util::geometry::demoscene,
    };

    use super::builder::build_bvh2_from_tris;

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
        let mut bvh = PlocBuilder::new().build(
            PlocSearchDistance::VeryLow,
            &aabbs,
            indices.clone(),
            SortPrecision::U64,
            1,
        );

        bvh.init_primitives_to_nodes_if_uninit();
        tris.transform(&Mat4::from_scale_rotation_translation(
            Vec3::splat(1.3),
            Quat::from_rotation_y(0.1),
            vec3(0.33, 0.3, 0.37),
        ));
        for (prim_id, tri) in tris.iter().enumerate() {
            bvh.nodes[bvh.primitives_to_nodes[prim_id] as usize].set_aabb(tri.aabb());
        }

        bvh.refit_all();

        bvh.validate(&tris, false, true);
    }

    #[test]
    fn test_reinsert_node() {
        let tris = demoscene(32, 0);

        let mut bvh = build_bvh2_from_tris(
            &tris,
            BvhBuildParams::fastest_build(),
            &mut Default::default(),
        );

        bvh.init_primitives_to_nodes_if_uninit();
        bvh.init_parents_if_uninit();

        for node_id in 1..bvh.nodes.len() {
            bvh.reinsert_node(node_id);
        }

        bvh.validate(&tris, false, false);
    }
}
