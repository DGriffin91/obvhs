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

const INVALID: u32 = u32::MAX;

/// A Compressed Wide BVH8
#[derive(Clone, Default, PartialEq)]
#[repr(C)]
pub struct CwBvh {
    pub nodes: Vec<CwBvhNode>,
    pub primitive_indices: Vec<u32>,
    pub total_aabb: Aabb,
}

const TRAVERSAL_STACK_SIZE: usize = 32;
#[derive(Default)]
pub struct TraversalStack32<T: Copy + Default> {
    data: [T; TRAVERSAL_STACK_SIZE],
    index: usize,
}

// TODO: check bounds in debug.
// TODO allow the user to provide their own stack impl via a Trait.
// BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no TLAS.
// A BVH that deep would perform very slowly and would likely indicate that the geometry is degenerate in some way.
// CwBvh::validate() will assert the CWBVH depth is less than TRAVERSAL_STACK_SIZE
impl<T: Copy + Default> TraversalStack32<T> {
    #[inline(always)]
    pub fn push(&mut self, v: T) {
        *unsafe { self.data.get_unchecked_mut(self.index) } = v;
        self.index = (self.index + 1).min(TRAVERSAL_STACK_SIZE - 1);
    }
    #[inline(always)]
    pub fn pop_fast(&mut self) -> T {
        self.index = self.index.saturating_sub(1);
        let v = *unsafe { self.data.get_unchecked(self.index) };
        v
    }
    #[inline(always)]
    pub fn pop(&mut self) -> Option<&T> {
        if self.index > 0 {
            self.index = self.index.saturating_sub(1);
            Some(&self.data[self.index])
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index == 0
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.index = 0;
    }
}

/// Holds traversal state to allow for dynamic traversal (yield on hit)
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
    /// For any hit, just run until the first time it yields true.
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
    /// reusing the same stack for both BLAS and TLAS traversal. It might be better to traverse separately
    /// on the CPU using two instances of `Traversal` with `CwBvh::traverse_dynamic()`. I haven't benchmarked this
    /// comparison yet. This example also does not take into account transforming the ray into the local
    /// space of the blas instance. (but has comments denoting where this would happen)
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
        self.validate_impl(0, Aabb::LARGEST, &mut ctx, primitives);
        //self.print_nodes();

        ctx.max_depth = self.caclulate_max_depth(0, &mut ctx, 0);

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
                // Empty
                continue;
            }

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

            let p = Vec3A::from(node.p);
            let quantized_min = quantized_min.as_vec3a() * e + p;
            let quantized_max = quantized_max.as_vec3a() * e + p;

            let is_child_inner = (node.imask & (1 << ch)) != 0;
            assert_eq!(is_child_inner, (child_meta & 0b11111) >= 24);

            if is_child_inner {
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
                ctx.leaf_count += 1;

                let first_prim = node.primitive_base_idx + (child_meta & 0b11111) as u32;
                for i in 0..3 {
                    if (child_meta & (0b1_00000 << i)) != 0 {
                        ctx.discovered_primitives.insert(first_prim + i);
                        ctx.prim_count += 1;
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
            dbg!(i);
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
