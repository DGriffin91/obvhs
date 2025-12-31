use glam::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    cwbvh::{
        CwBvhNode,
        node::{EPSILON, extract_byte64},
    },
    ray::Ray,
};

impl CwBvhNode {
    #[inline(always)]
    pub fn intersect_ray_simd(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adj_ray_dir_inv = self.compute_extent() * ray.inv_direction;
        let adj_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;
        let mut hit_mask = 0u32;
        unsafe {
            let adj_ray_dir_inv_x = _mm_set1_ps(adj_ray_dir_inv.x);
            let adj_ray_dir_inv_y = _mm_set1_ps(adj_ray_dir_inv.y);
            let adj_ray_dir_inv_z = _mm_set1_ps(adj_ray_dir_inv.z);

            let adj_ray_orig_x = _mm_set1_ps(adj_ray_origin.x);
            let adj_ray_orig_y = _mm_set1_ps(adj_ray_origin.y);
            let adj_ray_orig_z = _mm_set1_ps(adj_ray_origin.z);

            let rdx = ray.direction.x < 0.0;
            let rdy = ray.direction.y < 0.0;
            let rdz = ray.direction.z < 0.0;

            let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

            #[inline(always)]
            fn get_q(v: &[u8; 8], i: usize) -> __m128 {
                // get_q is the most expensive part of intersect_simd
                // Tried version with _mm_cvtepu8_epi32 and _mm_cvtepi32_ps, it was a lot slower.
                // Tried transmuting v into a u64 and bit shifting, it was a lot slower.
                unsafe {
                    _mm_set_ps(
                        *v.get_unchecked(i * 4 + 3) as f32,
                        *v.get_unchecked(i * 4 + 2) as f32,
                        *v.get_unchecked(i * 4 + 1) as f32,
                        *v.get_unchecked(i * 4) as f32,
                    )
                }
            }

            // Intersect 4 aabbs at a time:
            for i in 0..2 {
                // It's possible to select hi/lo outside the loop with child_min_x, etc... but that seems quite a bit slower
                // using _mm_blendv_ps or similar instead of `if rdx`, etc... is slower

                // Interleaving x, y, z like this is slightly faster than loading all at once. Tried using _mm_prefetch without luck
                let q_lo_x = get_q(&self.child_min_x, i);
                let q_hi_x = get_q(&self.child_max_x, i);
                let x_min = if rdx { q_hi_x } else { q_lo_x };
                let x_max = if rdx { q_lo_x } else { q_hi_x };
                // Tried using _mm_fmadd_ps, it was a lot slower
                let tmin_x = _mm_add_ps(_mm_mul_ps(x_min, adj_ray_dir_inv_x), adj_ray_orig_x);
                let tmax_x = _mm_add_ps(_mm_mul_ps(x_max, adj_ray_dir_inv_x), adj_ray_orig_x);

                let q_lo_y = get_q(&self.child_min_y, i);
                let q_hi_y = get_q(&self.child_max_y, i);
                let y_min = if rdy { q_hi_y } else { q_lo_y };
                let y_max = if rdy { q_lo_y } else { q_hi_y };
                let tmin_y = _mm_add_ps(_mm_mul_ps(y_min, adj_ray_dir_inv_y), adj_ray_orig_y);
                let tmax_y = _mm_add_ps(_mm_mul_ps(y_max, adj_ray_dir_inv_y), adj_ray_orig_y);

                let q_lo_z = get_q(&self.child_min_z, i);
                let q_hi_z = get_q(&self.child_max_z, i);
                let z_min = if rdz { q_hi_z } else { q_lo_z };
                let z_max = if rdz { q_lo_z } else { q_hi_z };
                let tmin_z = _mm_add_ps(_mm_mul_ps(z_min, adj_ray_dir_inv_z), adj_ray_orig_z);
                let tmax_z = _mm_add_ps(_mm_mul_ps(z_max, adj_ray_dir_inv_z), adj_ray_orig_z);

                // Tried using _mm_fmadd_ps, it was a lot slower
                // Compute intersection
                let tmin = _mm_max_ps(tmin_x, _mm_max_ps(tmin_y, tmin_z));
                let tmax = _mm_min_ps(tmax_x, _mm_min_ps(tmax_y, tmax_z));
                let tmin = _mm_max_ps(tmin, _mm_set1_ps(EPSILON)); //ray.tmin?
                let tmax = _mm_min_ps(tmax, _mm_set1_ps(ray.tmax));

                let intersected = _mm_cmple_ps(tmin, tmax);
                let mask = _mm_movemask_ps(intersected);

                for j in 0..4 {
                    let offset = i * 4 + j;
                    if (mask & (1 << j)) != 0 {
                        let child_bits = extract_byte64(child_bits8, offset);
                        let bit_index = extract_byte64(bit_index8, offset);
                        hit_mask |= child_bits << bit_index;
                    }
                }
            }
        }
        hit_mask
    }
}
