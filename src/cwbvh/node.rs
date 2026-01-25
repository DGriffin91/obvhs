use std::fmt::{self, Formatter};

use crate::{aabb::Aabb, ray::Ray};
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec3A, vec3a};
use std::fmt::Debug;

use super::NQ_SCALE;

/// A Compressed Wide BVH8 Node. repr(C), Pod, 80 bytes.
// https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CwBvhNode {
    /// Min point of node AABB
    pub p: Vec3,

    /// Exponent of child bounding box compression
    /// Max point of node AABB could be calculated ex: `p.x + bitcast<f32>(e[0] << 23) * 255.0`
    pub e: [u8; 3],

    /// Bitmask indicating which children are internal nodes. 1 for internal, 0 for leaf
    pub imask: u8,

    /// Index of first child into `Vec<CwBvhNode>`
    pub child_base_idx: u32,

    /// Index of first primitive into primitive_indices `Vec<u32>`
    pub primitive_base_idx: u32,

    /// Meta data for each child
    /// Empty child slot: The field is set to 00000000
    ///
    /// For leaves nodes: the low 5 bits store the primitive offset [0..24) from primitive_base_idx. And the high
    /// 3 bits store the number of primitives in that leaf in a unary encoding.
    /// A child leaf with 2 primitives with the first primitive starting at primitive_base_idx would be 0b01100000
    /// A child leaf with 3 primitives with the first primitive starting at primitive_base_idx + 2 would be 0b11100010
    /// A child leaf with 1 primitive with the first primitive starting at primitive_base_idx + 1 would be 0b00100001
    ///
    /// For internal nodes: The high 3 bits are set to 001 while the low 5 bits store the child slot index plus 24
    /// i.e., the values range [24..32)
    pub child_meta: [u8; 8],

    // Note: deviation from the paper: the min&max are interleaved here.
    /// Axis planes for each child.
    /// The plane position could be calculated, for example, with `p.x + bitcast<f32>(e[0] << 23) * child_min_x[0]`
    /// But in the actual intersection implementation the ray is transformed instead.
    pub child_min_x: [u8; 8],
    pub child_max_x: [u8; 8],
    pub child_min_y: [u8; 8],
    pub child_max_y: [u8; 8],
    pub child_min_z: [u8; 8],
    pub child_max_z: [u8; 8],
}

impl Debug for CwBvhNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CwBvhNode")
            .field("p", &self.p)
            .field("e", &self.e)
            .field("imask", &format!("{:#010b}", &self.imask))
            .field("child_base_idx", &self.child_base_idx)
            .field("primitive_base_idx", &self.primitive_base_idx)
            .field(
                "child_meta",
                &self
                    .child_meta
                    .iter()
                    .map(|c| format!("{c:#010b}"))
                    .collect::<Vec<_>>(),
            )
            .field("child_min_x", &self.child_min_x)
            .field("child_max_x", &self.child_max_x)
            .field("child_min_y", &self.child_min_y)
            .field("child_max_y", &self.child_max_y)
            .field("child_min_z", &self.child_min_z)
            .field("child_max_z", &self.child_max_z)
            .finish()
    }
}

pub(crate) const EPSILON: f32 = 0.0001;

impl CwBvhNode {
    #[inline(always)]
    pub fn intersect_ray(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ))]
        {
            self.intersect_ray_simd(ray, oct_inv4)
        }

        #[cfg(not(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        )))]
        {
            self.intersect_ray_basic(ray, oct_inv4)
        }
    }

    /// Intersects only one child at a time with the given ray. Limited simd usage on platforms that support it. Exists for reference & compatibility.
    /// Traversal times with CwBvhNode::intersect_ray_simd take less than half the time vs intersect_ray_basic.
    #[inline(always)]
    pub fn intersect_ray_basic(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adjusted_ray_dir_inv = self.compute_extent() * ray.inv_direction;
        let adjusted_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;

        let mut hit_mask = 0;

        let rdx = ray.direction.x < 0.0;
        let rdy = ray.direction.y < 0.0;
        let rdz = ray.direction.z < 0.0;

        let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

        for child in 0..8 {
            let q_lo_x = self.child_min_x[child];
            let q_lo_y = self.child_min_y[child];
            let q_lo_z = self.child_min_z[child];

            let q_hi_x = self.child_max_x[child];
            let q_hi_y = self.child_max_y[child];
            let q_hi_z = self.child_max_z[child];

            let x_min = if rdx { q_hi_x } else { q_lo_x };
            let x_max = if rdx { q_lo_x } else { q_hi_x };
            let y_min = if rdy { q_hi_y } else { q_lo_y };
            let y_max = if rdy { q_lo_y } else { q_hi_y };
            let z_min = if rdz { q_hi_z } else { q_lo_z };
            let z_max = if rdz { q_lo_z } else { q_hi_z };

            let mut tmin3 = vec3a(x_min as f32, y_min as f32, z_min as f32);
            let mut tmax3 = vec3a(x_max as f32, y_max as f32, z_max as f32);

            // Account for grid origin and scale
            tmin3 = tmin3 * adjusted_ray_dir_inv + adjusted_ray_origin;
            tmax3 = tmax3 * adjusted_ray_dir_inv + adjusted_ray_origin;

            let tmin = tmin3.max_element().max(EPSILON); //ray.tmin?
            let tmax = tmax3.min_element().min(ray.tmax);

            let intersected = tmin <= tmax;
            if intersected {
                let child_bits = extract_byte64(child_bits8, child);
                let bit_index = extract_byte64(bit_index8, child);
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn intersect_aabb(&self, aabb: &Aabb, oct_inv4: u32) -> u32 {
        let extent_rcp = 1.0 / self.compute_extent();
        let p = Vec3A::from(self.p);

        // Transform the query aabb into the node's local space
        let adjusted_aabb = Aabb::new((aabb.min - p) * extent_rcp, (aabb.max - p) * extent_rcp);

        let mut hit_mask = 0;

        let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

        for child in 0..8 {
            if self.local_child_aabb(child).intersect_aabb(&adjusted_aabb) {
                let child_bits = extract_byte64(child_bits8, child);
                let bit_index = extract_byte64(bit_index8, child);
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn contains_point(&self, point: &Vec3A, oct_inv4: u32) -> u32 {
        let extent_rcp = 1.0 / self.compute_extent();
        let p = Vec3A::from(self.p);

        // Transform the query point into the node's local space
        let adjusted_point = (*point - p) * extent_rcp;

        let mut hit_mask = 0;

        let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

        for child in 0..8 {
            if self.local_child_aabb(child).contains_point(adjusted_point) {
                let child_bits = extract_byte64(child_bits8, child);
                let bit_index = extract_byte64(bit_index8, child);
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    // TODO intersect frustum
    // https://github.com/zeux/niagara/blob/bf90aa8c78e352d3b753b35553a3bcc8c65ef7a0/src/shaders/drawcull.comp.glsl#L71
    // https://iquilezles.org/articles/frustumcorrect/

    #[inline(always)]
    pub fn get_child_and_index_bits(&self, oct_inv4: u32) -> (u64, u64) {
        let mut oct_inv8 = oct_inv4 as u64;
        oct_inv8 |= oct_inv8 << 32;
        let meta8 = u64::from_le_bytes(self.child_meta);

        // (meta8 & (meta8 << 1)) takes advantage of the offset indexing for inner nodes [24..32)
        // [0b00011000..=0b00011111). For leaf nodes [0..24) these two bits (0b00011000) are never both set.
        let inner_mask = 0b0001000000010000000100000001000000010000000100000001000000010000;
        let is_inner8 = (meta8 & (meta8 << 1)) & inner_mask;

        // 00010000 >> 4: 00000001, then 00000001 * 0xff: 11111111
        let inner_mask8 = (is_inner8 >> 4) * 0xffu64;

        // Each byte of bit_index8 contains the traversal priority, biased by 24, for internal nodes, and
        // the triangle offset for leaf nodes. The bit index will later be used to shift the child bits.
        let index_mask = 0b0001111100011111000111110001111100011111000111110001111100011111;
        let bit_index8 = (meta8 ^ (oct_inv8 & inner_mask8)) & index_mask;

        // For internal nodes child_bits8 will just be 1 in each byte, so that bit will then be shifted into the high
        // byte of the node hit_mask (see CwBvhNode::intersect_ray). For leaf nodes it will have the unary encoded
        // leaf primitive count and that will be shifted into the lower 24 bits of node hit_mask.
        let child_mask = 0b0000011100000111000001110000011100000111000001110000011100000111;
        let child_bits8 = (meta8 >> 5) & child_mask;
        (child_bits8, bit_index8)
    }

    /// Get local child aabb position relative to the parent
    #[inline(always)]
    pub fn local_child_aabb(&self, child: usize) -> Aabb {
        Aabb::new(
            vec3a(
                self.child_min_x[child] as f32,
                self.child_min_y[child] as f32,
                self.child_min_z[child] as f32,
            ),
            vec3a(
                self.child_max_x[child] as f32,
                self.child_max_y[child] as f32,
                self.child_max_z[child] as f32,
            ),
        )
    }

    #[inline(always)]
    pub fn child_aabb(&self, child: usize) -> Aabb {
        let e = self.compute_extent();
        let p: Vec3A = self.p.into();
        let mut local_aabb = self.local_child_aabb(child);
        local_aabb.min = local_aabb.min * e + p;
        local_aabb.max = local_aabb.max * e + p;
        local_aabb
    }

    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        let e = self.compute_extent();
        let p: Vec3A = self.p.into();
        Aabb::new(p, p + e * NQ_SCALE)
    }

    /// Convert stored extent exponent into float vector
    #[inline(always)]
    pub fn compute_extent(&self) -> Vec3A {
        vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        )
    }

    // If the child is empty this will also return true. If needed also use CwBvh::is_child_empty().
    #[inline(always)]
    pub fn is_leaf(&self, child: usize) -> bool {
        (self.imask & (1 << child)) == 0
    }

    #[inline(always)]
    pub fn is_child_empty(&self, child: usize) -> bool {
        self.child_meta[child] == 0
    }

    /// Returns the primitive starting index and primitive count for the given child.
    #[inline(always)]
    pub fn child_primitives(&self, child: usize) -> (u32, u32) {
        let child_meta = self.child_meta[child];
        let starting_index = self.primitive_base_idx + (self.child_meta[child] & 0b11111) as u32;
        let primitive_count = (child_meta & 0b11100000).count_ones();
        (starting_index, primitive_count)
    }

    /// Returns the node index of the given child.
    #[inline(always)]
    pub fn child_node_index(&self, child: usize) -> u32 {
        let child_meta = self.child_meta[child];
        let slot_index = (child_meta & 0b11111) as usize - 24;
        let relative_index = (self.imask as u32 & !(0xffffffffu32 << slot_index)).count_ones();
        self.child_base_idx + relative_index
    }
}

#[inline(always)]
pub fn extract_byte(x: u32, b: u32) -> u32 {
    (x >> (b * 8)) & 0xFFu32
}

#[inline(always)]
pub fn extract_byte64(x: u64, b: usize) -> u32 {
    ((x >> (b * 8)) as u32) & 0xFFu32
}
