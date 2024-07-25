//! An eight-way compressed wide BVH8 builder.

pub mod builder;
pub mod bvh2_to_cwbvh;
pub mod node;
pub mod simd;
pub mod traverse_macro;

use std::collections::{HashMap, HashSet};

use glam::{uvec2, UVec2, UVec3, Vec3A};
use node::CwBvhNode;

use crate::{
    aabb::Aabb,
    ray::{Ray, RayHit},
    Boundable, PerComponent,
};

pub const BRANCHING: usize = 8;

// Corresponds directly to the number of bit patterns created for child ordering
const DIRECTIONS: usize = 8;

const INVALID: u32 = u32::MAX;

const NQ: u32 = 8;
const NQ_SCALE: f32 = ((1 << NQ) - 1) as f32; //255.0
const DENOM: f32 = 1.0 / NQ_SCALE; // 1.0 / 255.0

/// A Compressed Wide BVH8
#[derive(Clone, Default, PartialEq, Debug)]
#[repr(C)]
pub struct CwBvh {
    pub nodes: Vec<CwBvhNode>,
    pub primitive_indices: Vec<u32>,
    pub total_aabb: Aabb,
    pub exact_node_aabbs: Option<Vec<Aabb>>,
}

const TRAVERSAL_STACK_SIZE: usize = 32;
/// A stack data structure implemented on the stack with fixed capacity.
#[derive(Default)]
pub struct TraversalStack32<T: Copy + Default> {
    data: [T; TRAVERSAL_STACK_SIZE],
    index: usize,
}

// TODO: possibly check bounds in debug.
// TODO allow the user to provide their own stack impl via a Trait.
// BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no TLAS.
// A BVH that deep would perform very slowly and would likely indicate that the geometry is degenerate in some way.
// CwBvh::validate() will assert the CWBVH depth is less than TRAVERSAL_STACK_SIZE
impl<T: Copy + Default> TraversalStack32<T> {
    /// Pushes a value onto the stack. If the stack is full it will overwrite the value in the last position.
    #[inline(always)]
    pub fn push(&mut self, v: T) {
        *unsafe { self.data.get_unchecked_mut(self.index) } = v;
        self.index = (self.index + 1).min(TRAVERSAL_STACK_SIZE - 1);
    }
    /// Pops a value from the stack without checking bounds. If the stack is empty it will return the value in the first position.
    #[inline(always)]
    pub fn pop_fast(&mut self) -> T {
        self.index = self.index.saturating_sub(1);
        let v = *unsafe { self.data.get_unchecked(self.index) };
        v
    }
    /// Pops a value from the stack.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<&T> {
        if self.index > 0 {
            self.index = self.index.saturating_sub(1);
            Some(&self.data[self.index])
        } else {
            None
        }
    }
    /// Returns the number of elements in the stack.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index
    }
    /// Returns true if the stack is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index == 0
    }
    /// Clears the stack, removing all elements.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.index = 0;
    }
}

/// Holds Ray traversal state to allow for dynamic traversal (yield on hit)
pub struct RayTraversal {
    pub stack: TraversalStack32<UVec2>,
    pub current_group: UVec2,
    pub primitive_group: UVec2,
    pub oct_inv4: u32,
    pub ray: Ray,
}

impl RayTraversal {
    #[inline(always)]
    /// Reinitialize traversal state with new ray.
    pub fn reinit(&mut self, ray: Ray) {
        self.stack.clear();
        self.current_group = uvec2(0, 0x80000000);
        self.primitive_group = UVec2::ZERO;
        self.oct_inv4 = ray_get_octant_inv4(&ray.direction);
        self.ray = ray;
    }
}

/// Holds traversal state to allow for dynamic traversal (yield on hit)
pub struct Traversal {
    pub stack: TraversalStack32<UVec2>,
    pub current_group: UVec2,
    pub primitive_group: UVec2,
    pub oct_inv4: u32,
    pub traversal_direction: Vec3A,
    pub primitive_id: u32,
    pub hitmask: u32,
}

impl Default for Traversal {
    fn default() -> Self {
        Self {
            stack: Default::default(),
            current_group: uvec2(0, 0x80000000),
            primitive_group: Default::default(),
            oct_inv4: Default::default(),
            traversal_direction: Default::default(),
            primitive_id: Default::default(),
            hitmask: Default::default(),
        }
    }
}

impl Traversal {
    #[inline(always)]
    /// Reinitialize traversal state with new traversal direction.
    pub fn reinit(&mut self, traversal_direction: Vec3A) {
        self.stack.clear();
        self.current_group = uvec2(0, 0x80000000);
        self.primitive_group = UVec2::ZERO;
        self.oct_inv4 = ray_get_octant_inv4(&traversal_direction);
        self.traversal_direction = traversal_direction;
        self.primitive_id = 0;
        self.hitmask = 0;
    }
}

impl CwBvh {
    #[inline(always)]
    pub fn new_ray_traversal(&self, ray: Ray) -> RayTraversal {
        //  BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no tlas.
        let stack = TraversalStack32::default();
        let current_group = if self.nodes.is_empty() {
            UVec2::ZERO
        } else {
            uvec2(0, 0x80000000)
        };
        let primitive_group = UVec2::ZERO;
        let oct_inv4 = ray_get_octant_inv4(&ray.direction);

        RayTraversal {
            stack,
            current_group,
            primitive_group,
            oct_inv4,
            ray,
        }
    }

    #[inline(always)]
    /// traversal_direction is used to determine the order of bvh node child traversal. This would typically be the ray direction.
    pub fn new_traversal(&self, traversal_direction: Vec3A) -> Traversal {
        //  BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no tlas.
        let stack = TraversalStack32::default();
        let current_group = if self.nodes.is_empty() {
            UVec2::ZERO
        } else {
            uvec2(0, 0x80000000)
        };
        let primitive_group = UVec2::ZERO;
        let oct_inv4 = ray_get_octant_inv4(&traversal_direction);
        Traversal {
            stack,
            current_group,
            primitive_group,
            oct_inv4,
            traversal_direction,
            primitive_id: 0,
            hitmask: 0,
        }
    }

    /// Traverse the BVH, finding the closest hit.
    /// Returns true if any primitive was hit.
    pub fn traverse<F: FnMut(&Ray, usize) -> f32>(
        &self,
        ray: Ray,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        let mut traverse_ray = ray.clone();
        let mut state = self.new_traversal(ray.direction);
        let mut node;
        crate::traverse!(
            self,
            node,
            state,
            node.intersect_ray(&traverse_ray, state.oct_inv4),
            {
                let t = intersection_fn(&traverse_ray, state.primitive_id as usize);
                if t < traverse_ray.tmax {
                    hit.primitive_id = state.primitive_id;
                    hit.t = t;
                    traverse_ray.tmax = t;
                }
            }
        );

        // Alternatively (performance seems slightly slower):
        // let mut state = self.new_ray_traversal(ray);
        // while self.traverse_dynamic(&mut state, hit, &mut intersection_fn) {}

        hit.t < ray.tmax // Note this is valid since traverse_dynamic does not mutate the ray
    }

    /// Traverse the BVH
    /// Yields at every primitive hit, returning true.
    /// Returns false when no hit is found.
    /// For basic miss test, just run until the first time it yields true.
    /// For closest hit run until it returns false and check hit.t < ray.tmax to see if it hit something
    /// For transparency, you want to hit every primitive in the ray's path, keeping track of the closest opaque hit.
    ///     and then manually setting ray.tmax to that closest opaque hit at each iteration.
    #[inline]
    pub fn traverse_dynamic<F: FnMut(&Ray, usize) -> f32>(
        &self,
        state: &mut RayTraversal,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        loop {
            // While the primitive group is not empty
            while state.primitive_group.y != 0 {
                let local_primitive_index = firstbithigh(state.primitive_group.y);

                // Remove primitive from current_group
                state.primitive_group.y &= !(1u32 << local_primitive_index);

                let global_primitive_index = state.primitive_group.x + local_primitive_index;
                let t = intersection_fn(&state.ray, global_primitive_index as usize);
                if t < state.ray.tmax {
                    hit.primitive_id = global_primitive_index;
                    hit.t = t;
                    state.ray.tmax = hit.t;
                    // Yield when we hit a primitive
                    return true;
                }
            }
            state.primitive_group = UVec2::ZERO;

            // If there's remaining nodes in the current group to check
            if state.current_group.y & 0xff000000 != 0 {
                let hits_imask = state.current_group.y;

                let child_index_offset = firstbithigh(hits_imask);
                let child_index_base = state.current_group.x;

                // Remove node from current_group
                state.current_group.y &= !(1u32 << child_index_offset);

                // If the node group is not yet empty, push it on the stack
                if state.current_group.y & 0xff000000 != 0 {
                    state.stack.push(state.current_group);
                }

                let slot_index = (child_index_offset - 24) ^ (state.oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();

                let child_node_index = child_index_base + relative_index;

                let node = &self.nodes[child_node_index as usize];

                let hitmask = node.intersect_ray(&state.ray, state.oct_inv4);

                state.current_group.x = node.child_base_idx;
                state.primitive_group.x = node.primitive_base_idx;

                state.current_group.y = (hitmask & 0xff000000) | (node.imask as u32);
                state.primitive_group.y = hitmask & 0x00ffffff;
            } else
            // There's no nodes left in the current group
            {
                // Below is only needed when using triangle postponing, which would only be helpful on the
                // GPU (it helps reduce thread divergence). Also, this isn't compatible with traversal yeilding.
                // state.primitive_group = state.current_group;
                state.current_group = UVec2::ZERO;
            }

            // If there's no remaining nodes in the current group to check, pop it off the stack.
            if state.primitive_group.y == 0 && (state.current_group.y & 0xff000000) == 0 {
                // If the stack is empty, end traversal.
                if state.stack.is_empty() {
                    state.current_group.y = 0;
                    break;
                }

                state.current_group = state.stack.pop_fast();
            }
        }

        // Returns false when there are no more primitives to test.
        // This doesn't mean we never hit one along the way though. (and yielded then)
        false
    }

    /// This is currently mostly here just for reference. It's setup somewhat similarly to the GPU version,
    /// reusing the same stack for both BLAS and TLAS traversal. It might be better to traverse separately on
    /// the CPU using two instances of `Traversal` with `CwBvh::traverse_dynamic()` or the `traverse!` macro.
    /// I haven't benchmarked this comparison yet. This example also does not take into account transforming
    /// the ray into the local space of the blas instance. (but has comments denoting where this would happen)
    pub fn traverse_tlas_blas<F: FnMut(&Ray, usize, usize) -> f32>(
        &self,
        blas: &[CwBvh],
        mut ray: Ray,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        let mut stack = TraversalStack32::default();
        let mut current_group;
        let mut tlas_stack_size = INVALID; // tlas_stack_size is used to indicate whether we are in the TLAS or not.
        let mut current_mesh = INVALID;
        let mut bvh = self;

        let oct_inv4 = ray_get_octant_inv4(&ray.direction);

        current_group = uvec2(0, 0x80000000);

        loop {
            let mut primitive_group = UVec2::ZERO;

            // If there's remaining nodes in the current group to check
            if current_group.y & 0xff000000 != 0 {
                let hits_imask = current_group.y;

                let child_index_offset = firstbithigh(hits_imask);
                let child_index_base = current_group.x;

                // Remove node from current_group
                current_group.y &= !(1u32 << child_index_offset);

                // If the node group is not yet empty, push it on the stack
                if current_group.y & 0xff000000 != 0 {
                    stack.push(current_group);
                }

                let slot_index = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();

                let child_node_index = child_index_base + relative_index;

                let node = &bvh.nodes[child_node_index as usize];

                let hitmask = node.intersect_ray(&ray, oct_inv4);

                current_group.x = node.child_base_idx;
                primitive_group.x = node.primitive_base_idx;

                current_group.y = (hitmask & 0xff000000) | (node.imask as u32);
                primitive_group.y = hitmask & 0x00ffffff;
            } else
            // There's no nodes left in the current group
            {
                // Below is only needed when using triangle postponing, which would only be helpful on the
                // GPU (it helps reduce thread divergence). Also, this isn't compatible with traversal yeilding.
                // primitive_group = current_group;
                current_group = UVec2::ZERO;
            }

            // While the primitive group is not empty
            while primitive_group.y != 0 {
                // https://github.com/jan-van-bergen/GPU-Raytracer/issues/24#issuecomment-1042746566
                // If tlas_stack_size is INVALID we are in the TLAS. This means use the primitive index as a mesh index.
                // (TODO: The ray is transform according to the mesh transform and) traversal is continued at the root of the Mesh's BLAS.
                if tlas_stack_size == INVALID {
                    let local_primitive_index = firstbithigh(primitive_group.y);

                    // Remove primitive from current_group
                    primitive_group.y &= !(1u32 << local_primitive_index);

                    // Mesh id or entitiy id. If entitiy, it would be necessary to look up mesh id from entity.
                    let global_primitive_index = primitive_group.x + local_primitive_index;

                    if primitive_group.y != 0 {
                        stack.push(primitive_group);
                    }

                    if current_group.y & 0xff000000 != 0 {
                        stack.push(current_group);
                    }

                    // The value of tlas_stack_size is now set to the current size of the traversal stack.
                    tlas_stack_size = stack.len() as u32;

                    // TODO transform ray according to the mesh transform
                    // https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/CUDA/Raytracing/BVH8.h#L222

                    // For primitives, we remap them to match the cwbvh indices layout. But for tlas
                    // it would not be typically reasonable to reorder the blas and mesh buffers. So we
                    // need to look up the original index using bvh.primitive_indices[].
                    let blas_index = bvh.primitive_indices[global_primitive_index as usize];
                    bvh = &blas[blas_index as usize];
                    current_mesh = blas_index;

                    // since we assign bvh = &blas[global_primitive_index as usize] above the index is just the first node at 0
                    current_group = uvec2(0, 0x80000000);

                    break;
                } else {
                    // If tlas_stack_size is any other value we are in the BLAS. This performs the usual primitive intersection.

                    let local_primitive_index = firstbithigh(primitive_group.y);

                    // Remove primitive from current_group
                    primitive_group.y &= !(1u32 << local_primitive_index);

                    let global_primitive_index = primitive_group.x + local_primitive_index;
                    let t = intersection_fn(
                        &ray,
                        current_mesh as usize,
                        global_primitive_index as usize,
                    );

                    if t < ray.tmax {
                        hit.primitive_id = global_primitive_index;
                        hit.geometry_id = current_mesh;
                        ray.tmax = t;
                    }
                }
            }

            // If there's no remaining nodes in the current group to check, pop it off the stack.
            if (current_group.y & 0xff000000) == 0 {
                // If the stack is empty, end traversal.
                if stack.is_empty() {
                    current_group.y = 0;
                    break;
                }

                // The value of tlas_stack_size is used to determine when traversal of a BLAS is finished, and we should revert back to TLAS traversal.
                if stack.len() as u32 == tlas_stack_size {
                    tlas_stack_size = INVALID;
                    current_mesh = INVALID;
                    bvh = self;
                    // TODO Reset Ray to untransformed version
                    // https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/CUDA/Raytracing/BVH8.h#L262
                }

                current_group = stack.pop_fast();
            }
        }

        if hit.primitive_id != u32::MAX {
            hit.t = ray.tmax;
            return true;
        }

        false
    }

    /// Returns the list of parents where `parent_index = parents[node_index]`
    pub fn compute_parents(&self) -> Vec<u32> {
        let mut parents = vec![0; self.nodes.len()];
        parents[0] = 0;
        self.nodes.iter().enumerate().for_each(|(i, node)| {
            for ch in 0..8 {
                if node.is_child_empty(ch) {
                    continue;
                }
                if !node.is_leaf(ch) {
                    parents[node.child_node_index(ch) as usize] = i as u32;
                }
            }
        });
        parents
    }

    /// Reorder the children of every BVH node. This results in a slightly different order since the normal reordering during
    /// building is using the aabb's from the BVH2 and this uses the children node.p and node.e to compute the aabb. Traversal
    /// seems to be a bit slower on some scenes and a bit faster on others. Note this will rearrange self.nodes. Anything that
    /// depends on the order of self.nodes will need to be updated.
    ///
    /// # Arguments
    /// * `direct_layout` - The primitives are already laid out in bvh.primitive_indices order.
    /// * `primitives` - List of BVH primitives, implementing Boundable.
    pub fn order_children<T: Boundable>(&mut self, direct_layout: bool, primitives: &[T]) {
        for i in 0..self.nodes.len() {
            self.order_node_children(i, direct_layout, primitives);
        }
    }

    /// Reorder the children of the given node_idx. This results in a slightly different order since the normal reordering during
    /// building is using the aabb's from the BVH2 and this uses the children node.p and node.e to compute the aabb. Traversal
    /// seems to be a bit slower on some scenes and a bit faster on others. Note this will rearrange self.nodes. Anything that
    /// depends on the order of self.nodes will need to be updated.
    ///
    /// # Arguments
    /// * `node_idx` - Index of node to be reordered.
    /// * `direct_layout` - The primitives are already laid out in bvh.primitive_indices order.
    /// * `primitives` - List of BVH primitives, implementing Boundable.
    pub fn order_node_children<T: Boundable>(
        &mut self,
        node_index: usize,
        direct_layout: bool,
        primitives: &[T],
    ) {
        // TODO could this use ints and work in local node grid space?

        let old_node = self.nodes[node_index];

        const INVALID32: u32 = u32::MAX;
        const INVALID_USIZE: usize = INVALID32 as usize;
        let center = old_node.aabb().center();

        let mut cost = [[f32::MAX; DIRECTIONS]; BRANCHING];

        let mut child_count = 0;
        let mut child_inner_count = 0;
        for ch in 0..BRANCHING {
            if !old_node.is_child_empty(ch) {
                child_count += 1;
                if !old_node.is_leaf(ch) {
                    child_inner_count += 1;
                }
            }
        }

        let mut old_child_centers = [Vec3A::default(); 8];
        for ch in 0..BRANCHING {
            if old_node.is_child_empty(ch) {
                continue;
            }
            if old_node.is_leaf(ch) {
                let (child_prim_start, count) = old_node.child_primitives(ch);
                let mut aabb = Aabb::empty();
                for i in 0..count {
                    let mut prim_index = (child_prim_start + i) as usize;
                    if !direct_layout {
                        prim_index = self.primitive_indices[prim_index] as usize;
                    }
                    aabb = aabb.union(&primitives[prim_index].aabb());
                }
                old_child_centers[ch] = aabb.center();
            } else {
                old_child_centers[ch] = self.nodes[old_node.child_node_index(ch) as usize]
                    .aabb()
                    .center();
                let child_node_index = old_node.child_node_index(ch) as usize;
                old_child_centers[ch] = self.node_aabb(child_node_index).center();
            }
        }

        assert!(child_count <= BRANCHING);
        assert!(cost.len() >= child_count);
        // Fill cost table
        // TODO parallel: check to see if this is faster w/ par_iter
        for s in 0..DIRECTIONS {
            let d = Vec3A::new(
                if (s & 0b100) != 0 { -1.0 } else { 1.0 },
                if (s & 0b010) != 0 { -1.0 } else { 1.0 },
                if (s & 0b001) != 0 { -1.0 } else { 1.0 },
            );
            // We have to use BRANCHING here instead of child_count because the first slots wont be children if it was already reordered.
            for ch in 0..BRANCHING {
                if old_node.is_child_empty(ch) {
                    continue;
                }
                let v = old_child_centers[ch] - center; //old_node.child_aabb(c).center() - center;
                let cost_slot = unsafe { cost.get_unchecked_mut(ch).get_unchecked_mut(s) };
                *cost_slot = d.dot(v); // No benefit from normalizing
            }
        }

        let mut assignment = [INVALID_USIZE; BRANCHING];
        let mut slot_filled = [false; DIRECTIONS];

        // The paper suggests the auction method, but greedy is almost as good.
        loop {
            let mut min_cost = f32::MAX;

            let mut min_slot = INVALID_USIZE;
            let mut min_index = INVALID_USIZE;

            // Find cheapest unfilled slot of any unassigned child
            // We have to use BRANCHING here instead of child_count because the first slots wont be children if it was already reordered.
            for ch in 0..BRANCHING {
                if old_node.is_child_empty(ch) {
                    continue;
                }
                if assignment[ch] == INVALID_USIZE {
                    for (s, &slot_filled) in slot_filled.iter().enumerate() {
                        let cost = unsafe { *cost.get_unchecked(ch).get_unchecked(s) };
                        if !slot_filled && cost < min_cost {
                            min_cost = cost;

                            min_slot = s;
                            min_index = ch;
                        }
                    }
                }
            }

            if min_slot == INVALID_USIZE {
                break;
            }

            slot_filled[min_slot] = true;
            assignment[min_index] = min_slot;
        }

        let mut new_node = old_node.clone();
        new_node.imask = 0;

        for ch in 0..BRANCHING {
            new_node.child_meta[ch] = 0;
        }

        for ch in 0..BRANCHING {
            if old_node.is_child_empty(ch) {
                continue;
            }
            let new_ch = assignment[ch];
            assert!(new_ch < BRANCHING);
            if old_node.is_leaf(ch) {
                new_node.child_meta[new_ch] = old_node.child_meta[ch];
            } else {
                new_node.imask |= 1 << new_ch;
                new_node.child_meta[new_ch] = (24 + new_ch as u8) | 0b0010_0000;
            }
            new_node.child_min_x[new_ch] = old_node.child_min_x[ch];
            new_node.child_max_x[new_ch] = old_node.child_max_x[ch];
            new_node.child_min_y[new_ch] = old_node.child_min_y[ch];
            new_node.child_max_y[new_ch] = old_node.child_max_y[ch];
            new_node.child_min_z[new_ch] = old_node.child_min_z[ch];
            new_node.child_max_z[new_ch] = old_node.child_max_z[ch];
        }

        if child_inner_count == 0 {
            self.nodes[node_index] = new_node;
            return;
        }

        let mut old_child_nodes = [CwBvhNode::default(); 8];
        for ch in 0..BRANCHING {
            if old_node.is_child_empty(ch) {
                continue;
            }
            if old_node.is_leaf(ch) {
                continue;
            }
            old_child_nodes[ch] = self.nodes[old_node.child_node_index(ch) as usize]
        }

        let old_child_exact_aabbs = if let Some(exact_node_aabbs) = &self.exact_node_aabbs {
            let mut old_child_exact_aabbs = [Aabb::empty(); 8];
            for ch in 0..BRANCHING {
                if old_node.is_child_empty(ch) {
                    continue;
                }
                if old_node.is_leaf(ch) {
                    continue;
                }
                old_child_exact_aabbs[ch] =
                    exact_node_aabbs[old_node.child_node_index(ch) as usize];
            }
            Some(old_child_exact_aabbs)
        } else {
            None
        };

        // check if this is really needed or if we can specify the offset in the child_meta out of order
        for ch in 0..BRANCHING {
            if old_node.is_child_empty(ch) {
                continue;
            }
            if assignment[ch] == INVALID_USIZE {
                continue;
            }
            let new_ch = assignment[ch];
            assert_eq!(
                !new_node.is_leaf(new_ch),
                (new_node.child_meta[new_ch] & 0b11111) >= 24
            );
            if old_node.is_leaf(ch) {
                continue;
            }
            let new_idx = new_node.child_node_index(new_ch) as usize;
            self.nodes[new_idx] = old_child_nodes[ch];
            if let Some(old_child_exact_aabbs) = &old_child_exact_aabbs {
                if let Some(exact_node_aabbs) = &mut self.exact_node_aabbs {
                    exact_node_aabbs[new_idx] = old_child_exact_aabbs[ch];
                }
            }
            assert!(new_idx >= old_node.child_base_idx as usize);
            assert!(new_idx < old_node.child_base_idx as usize + child_inner_count);
        }
        self.nodes[node_index] = new_node;
    }

    /// Tries to use the exact node aabb if it is available, otherwise computes it from the compressed node min P and extent exponent.
    #[inline(always)]
    fn node_aabb(&self, node_index: usize) -> Aabb {
        if let Some(exact_node_aabbs) = &self.exact_node_aabbs {
            exact_node_aabbs[node_index]
        } else {
            self.nodes[node_index].aabb()
        }
    }

    /// Direct layout: The primitives are already laid out in bvh.primitive_indices order.
    pub fn validate<T: Boundable>(
        &self,
        splits: bool,
        direct_layout: bool,
        primitives: &[T],
    ) -> CwBvhValidateCtx {
        if !splits {
            // Could still check this if duplicated were removed from self.primitive_indices first
            assert_eq!(self.primitive_indices.len(), primitives.len());
        }
        let mut ctx = CwBvhValidateCtx {
            splits,
            direct_layout,
            ..Default::default()
        };
        if !self.nodes.is_empty() {
            self.validate_impl(0, Aabb::LARGEST, &mut ctx, primitives);
        }
        //self.print_nodes();

        ctx.max_depth = self.caclulate_max_depth(0, &mut ctx, 0);

        if let Some(exact_node_aabbs) = &self.exact_node_aabbs {
            for node in &self.nodes {
                for ch in 0..8 {
                    if !node.is_leaf(ch) {
                        let child_node_index = node.child_node_index(ch) as usize;
                        let comp_aabb = node.child_aabb(ch);
                        let self_aabb = self.nodes[child_node_index].aabb();
                        let exact_aabb = exact_node_aabbs[child_node_index];

                        // TODO Could these bounds be tighter?
                        assert!(exact_aabb.min.cmpge(comp_aabb.min - 1.0e-5).all());
                        assert!(exact_aabb.max.cmple(comp_aabb.max + 1.0e-5).all());
                        assert!(exact_aabb.min.cmpge(self_aabb.min - 1.0e-5).all());
                        assert!(exact_aabb.max.cmple(self_aabb.max + 1.0e-5).all());
                    }
                }
            }
        }

        assert_eq!(ctx.discovered_nodes.len(), self.nodes.len());
        assert_eq!(
            ctx.discovered_primitives.len(),
            self.primitive_indices.len()
        );
        assert!(ctx.max_depth < TRAVERSAL_STACK_SIZE);

        ctx
    }

    fn validate_impl<T: Boundable>(
        &self,
        node_idx: usize,
        parent_bounds: Aabb,
        ctx: &mut CwBvhValidateCtx,
        primitives: &[T],
    ) {
        ctx.discovered_nodes.insert(node_idx as u32);
        ctx.node_count += 1;

        let node = &self.nodes[node_idx];

        assert!(node.p.is_finite());
        assert!(parent_bounds.min.is_finite());
        assert!(parent_bounds.max.is_finite());
        // TODO Could these bounds be tighter?
        assert!(node.p.cmpge((parent_bounds.min - 1.0e-5).into()).all());
        assert!(node.p.cmple((parent_bounds.max + 1.0e-5).into()).all());

        let e: UVec3 = [
            (node.e[0] as u32) << 23,
            (node.e[1] as u32) << 23,
            (node.e[2] as u32) << 23,
        ]
        .into();
        let e: Vec3A = e.per_comp(f32::from_bits);

        for ch in 0..8 {
            let child_meta = node.child_meta[ch];
            if child_meta == 0 {
                assert!(node.is_child_empty(ch));
                // Empty
                continue;
            }
            assert!(!node.is_child_empty(ch));

            ctx.child_count += 1;

            let quantized_min = UVec3::new(
                node.child_min_x[ch] as u32,
                node.child_min_y[ch] as u32,
                node.child_min_z[ch] as u32,
            );
            let quantized_max = UVec3::new(
                node.child_max_x[ch] as u32,
                node.child_max_y[ch] as u32,
                node.child_max_z[ch] as u32,
            );

            assert_eq!(
                Aabb::new(quantized_min.as_vec3a(), quantized_max.as_vec3a()),
                node.local_child_aabb(ch)
            );

            let p = Vec3A::from(node.p);
            let quantized_min = quantized_min.as_vec3a() * e + p;
            let quantized_max = quantized_max.as_vec3a() * e + p;

            assert_eq!(Aabb::new(quantized_min, quantized_max), node.child_aabb(ch));

            let is_child_inner = (node.imask & (1 << ch)) != 0;
            assert_eq!(is_child_inner, (child_meta & 0b11111) >= 24);

            if is_child_inner {
                assert!(!node.is_leaf(ch));
                let slot_index = (child_meta & 0b11111) as usize - 24;
                let relative_index =
                    (node.imask as u32 & !(0xffffffffu32 << slot_index)).count_ones();
                let child_node_idx = node.child_base_idx as usize + relative_index as usize;
                self.validate_impl(
                    child_node_idx,
                    Aabb {
                        min: quantized_min,
                        max: quantized_max,
                    }
                    .intersection(&parent_bounds),
                    ctx,
                    primitives,
                );
            } else {
                assert!(node.is_leaf(ch));
                ctx.leaf_count += 1;

                let first_prim = node.primitive_base_idx + (child_meta & 0b11111) as u32;
                assert_eq!(first_prim, node.child_primitives(ch).0);
                let mut prim_count = 0;
                for i in 0..3 {
                    if (child_meta & (0b1_00000 << i)) != 0 {
                        ctx.discovered_primitives.insert(first_prim + i);
                        ctx.prim_count += 1;
                        prim_count += 1;
                        let mut prim_index = (first_prim + i) as usize;
                        if !ctx.direct_layout {
                            prim_index = self.primitive_indices[prim_index] as usize;
                        }
                        let prim_aabb = primitives[prim_index].aabb();

                        if !ctx.splits {
                            // TODO: option that correctly takes into account error of compressed triangle.
                            // Maybe Boundable can return an epsilon, and for compressed triangles it
                            // can take into account the edge length
                            assert!(
                                prim_aabb.min.cmpge(parent_bounds.min - 1.0e-5).all()
                                    && prim_aabb.max.cmple(parent_bounds.max + 1.0e-5).all(),
                                "Primitive {} does not fit in parent {}:\nprimitive: {:?}\nparent:    {:?}",
                                prim_index,
                                node_idx,
                                prim_aabb,
                                parent_bounds
                            );
                        }
                    }
                }
                assert_eq!(prim_count, node.child_primitives(ch).1);
            }
        }
    }

    /// Calculate the maximum depth of the BVH from this node down.
    fn caclulate_max_depth(
        &self,
        node_idx: usize,
        ctx: &mut CwBvhValidateCtx,
        current_depth: usize,
    ) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let node = &self.nodes[node_idx];
        let mut max_depth = current_depth;

        if let Some(count) = ctx.nodes_at_depth.get(&current_depth) {
            ctx.nodes_at_depth.insert(current_depth, count + 1);
        } else {
            ctx.nodes_at_depth.insert(current_depth, 1);
        }

        for ch in 0..8 {
            let child_meta = node.child_meta[ch];
            if child_meta == 0 {
                // Empty
                continue;
            }

            let is_child_inner = (node.imask & (1 << ch)) != 0;
            assert_eq!(is_child_inner, (child_meta & 0b11111) >= 24);

            if is_child_inner {
                let slot_index = (child_meta & 0b11111) as usize - 24;
                let relative_index =
                    (node.imask as u32 & !(0xffffffffu32 << slot_index)).count_ones();
                let child_node_idx = node.child_base_idx as usize + relative_index as usize;

                let child_depth = self.caclulate_max_depth(child_node_idx, ctx, current_depth + 1);

                max_depth = max_depth.max(child_depth);
            } else {
                // Leaf
                // max_depth = max_depth.max(current_depth + 1);

                if let Some(count) = ctx.leaves_at_depth.get(&current_depth) {
                    ctx.leaves_at_depth.insert(current_depth, count + 1);
                } else {
                    ctx.leaves_at_depth.insert(current_depth, 1);
                }
            }
        }

        max_depth
    }

    #[allow(dead_code)]
    fn print_nodes(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!("node: {}", i);
            for ch in 0..8 {
                let child_meta = node.child_meta[ch];
                if child_meta == 0 {
                    // Empty
                    continue;
                }

                let is_child_inner = (node.imask & (1 << ch)) != 0;
                assert_eq!(is_child_inner, (child_meta & 0b11111) >= 24);

                if is_child_inner {
                    println!("inner");
                } else {
                    // Leaf
                    let mut prims = 0;
                    for i in 0..3 {
                        if (child_meta & (0b1_00000 << i)) != 0 {
                            prims += 1;
                        }
                    }
                    println!("leaf, prims: {}", prims);
                }
            }
        }
    }
}

#[inline(always)]
pub fn firstbithigh(value: u32) -> u32 {
    31 - value.leading_zeros()
}

#[inline(always)]
fn ray_get_octant_inv4(dir: &Vec3A) -> u32 {
    // Ray octant, encoded in 3 bits
    // let oct = (if dir.x < 0.0 { 0b100 } else { 0 })
    //     | (if dir.y < 0.0 { 0b010 } else { 0 })
    //     | (if dir.z < 0.0 { 0b001 } else { 0 });
    // return (7 - oct) * 0x01010101;
    (if dir.x < 0.0 { 0 } else { 0x04040404 }
        | if dir.y < 0.0 { 0 } else { 0x02020202 }
        | if dir.z < 0.0 { 0 } else { 0x01010101 })
}

#[derive(Default)]
pub struct CwBvhValidateCtx {
    pub splits: bool,
    pub direct_layout: bool,
    pub discovered_primitives: HashSet<u32>,
    pub discovered_nodes: HashSet<u32>,
    pub node_count: usize,
    pub child_count: usize,
    pub leaf_count: usize,
    pub prim_count: usize,
    pub max_depth: usize,
    pub nodes_at_depth: HashMap<usize, usize>,
    pub leaves_at_depth: HashMap<usize, usize>,
}

impl CwBvhValidateCtx {
    pub fn print(&self) {
        println!(
            "GPU BVH Avg children/node: {:.3}, primitives/leaf: {:.3}",
            self.child_count as f64 / self.node_count as f64,
            self.prim_count as f64 / self.leaf_count as f64
        );

        println!(
            "\
child_count: {}
 node_count: {}
 prim_count: {}
 leaf_count: {}",
            self.child_count, self.node_count, self.prim_count, self.leaf_count
        );

        println!("Node & Leaf counts for each depth");
        for i in 0..=self.max_depth {
            println!(
                "{:<3} {:<10} {:<10}",
                i,
                self.nodes_at_depth.get(&i).unwrap_or(&0),
                self.leaves_at_depth.get(&i).unwrap_or(&0)
            );
        }
    }
}
