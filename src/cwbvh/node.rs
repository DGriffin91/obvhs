use bytemuck::{Pod, Zeroable};
use glam::{vec3a, Vec3, Vec3A};

use crate::{aabb::Aabb, ray::Ray};

/// A Compressed Wide BVH8 Node
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct CwBvhNode {
    /// Min point
    pub p: Vec3,
    /// Exponent of child bounding box compression
    pub e: [u8; 3],
    /// Indicates which children are internal nodes
    pub imask: u8,
    pub child_base_idx: u32,
    pub primitive_base_idx: u32,
    pub child_meta: [u8; 8],

    // Note: deviation from the paper: the min&max are interleaved here.
    pub child_min_x: [u8; 8],
    pub child_max_x: [u8; 8],
    pub child_min_y: [u8; 8],
    pub child_max_y: [u8; 8],
    pub child_min_z: [u8; 8],
    pub child_max_z: [u8; 8],
}

pub(crate) const EPSILON: f32 = 0.0001;
unsafe impl Pod for CwBvhNode {}
unsafe impl Zeroable for CwBvhNode {}

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
    #[inline(always)]
    pub fn intersect_ray_basic(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adjusted_ray_dir_inv = vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        ) * ray.inv_direction;
        let adjusted_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;

        let mut hit_mask = 0;

        let rdx = ray.direction.x < 0.0;
        let rdy = ray.direction.y < 0.0;
        let rdz = ray.direction.z < 0.0;

        // [unroll]
        for i in 0..2 {
            let meta4 = extract_u32(&self.child_meta, i == 0);
            let is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
            let inner_mask4 = (is_inner4 >> 4) * 0xffu32;
            let bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
            let child_bits4 = (meta4 >> 5) & 0x07070707;

            // [unroll]
            for j in 0..4 {
                let ch = i * 4 + j;

                let q_lo_x = self.child_min_x[ch];
                let q_lo_y = self.child_min_y[ch];
                let q_lo_z = self.child_min_z[ch];

                let q_hi_x = self.child_max_x[ch];
                let q_hi_y = self.child_max_y[ch];
                let q_hi_z = self.child_max_z[ch];

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

                let tmin = tmin3.x.max(tmin3.y).max(tmin3.z).max(EPSILON); //ray.tmin?
                let tmax = tmax3.x.min(tmax3.y).min(tmax3.z).min(ray.tmax);

                let intersected = tmin <= tmax;
                if intersected {
                    let child_bits = extract_byte(child_bits4, j as u32);
                    let bit_index = extract_byte(bit_index4, j as u32);

                    hit_mask |= child_bits << bit_index;
                }
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn intersect_aabb(&self, aabb: &Aabb, oct_inv4: u32) -> u32 {
        let extent = vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        );
        let extent_rcp = 1.0 / extent;
        let p = Vec3A::from(self.p);

        // Transform the query aabb into the node's local space
        let adjusted_aabb = Aabb::new((aabb.min - p) * extent_rcp, (aabb.max - p) * extent_rcp);

        let mut hit_mask = 0;

        for i in 0..2 {
            let meta4 = extract_u32(&self.child_meta, i == 0);
            let is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
            let inner_mask4 = (is_inner4 >> 4) * 0xffu32;
            let bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
            let child_bits4 = (meta4 >> 5) & 0x07070707;

            for j in 0..4 {
                let ch = i * 4 + j;
                let child_aabb = Aabb::new(
                    vec3a(
                        self.child_min_x[ch] as f32,
                        self.child_min_y[ch] as f32,
                        self.child_min_z[ch] as f32,
                    ),
                    vec3a(
                        self.child_max_x[ch] as f32,
                        self.child_max_y[ch] as f32,
                        self.child_max_z[ch] as f32,
                    ),
                );

                if child_aabb.intersect_aabb(&adjusted_aabb) {
                    let child_bits = extract_byte(child_bits4, j as u32);
                    let bit_index = extract_byte(bit_index4, j as u32);

                    hit_mask |= child_bits << bit_index;
                }
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn contains_point(&self, point: &Vec3A, oct_inv4: u32) -> u32 {
        let extent = vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        );
        let extent_rcp = 1.0 / extent;
        let p = Vec3A::from(self.p);

        // Transform the query point into the node's local space
        let adjusted_point = (*point - p) * extent_rcp;

        let mut hit_mask = 0;

        for i in 0..2 {
            let meta4 = extract_u32(&self.child_meta, i == 0);
            let is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
            let inner_mask4 = (is_inner4 >> 4) * 0xffu32;
            let bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
            let child_bits4 = (meta4 >> 5) & 0x07070707;

            for j in 0..4 {
                let ch = i * 4 + j;
                let child_aabb = Aabb::new(
                    vec3a(
                        self.child_min_x[ch] as f32,
                        self.child_min_y[ch] as f32,
                        self.child_min_z[ch] as f32,
                    ),
                    vec3a(
                        self.child_max_x[ch] as f32,
                        self.child_max_y[ch] as f32,
                        self.child_max_z[ch] as f32,
                    ),
                );

                if child_aabb.contains_point(adjusted_point) {
                    let child_bits = extract_byte(child_bits4, j as u32);
                    let bit_index = extract_byte(bit_index4, j as u32);

                    hit_mask |= child_bits << bit_index;
                }
            }
        }

        hit_mask
    }
}

#[inline(always)]
#[allow(dead_code)]
pub fn extract_byte(x: u32, b: u32) -> u32 {
    (x >> (b * 8)) & 0xFFu32
}

#[inline(always)]
#[allow(dead_code)]
pub fn extract_byte64(x: u64, b: usize) -> u32 {
    ((x >> (b * 8)) as u32) & 0xFFu32
}

#[inline(always)]
pub fn extract_u32(data: &[u8; 8], second: bool) -> u32 {
    unsafe { *(data.as_ptr().add(if second { 0 } else { 4 }) as *const u32) }
}
