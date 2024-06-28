use glam::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

use crate::{
    cwbvh::{node::EPSILON, CwBvhNode},
    ray::Ray,
};

impl CwBvhNode {
    #[inline(always)]
    pub fn intersect_ray_simd(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adj_ray_dir_inv = vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        ) * ray.inv_direction;
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

            let mut oct_inv8 = oct_inv4 as u64;
            oct_inv8 |= oct_inv8 << 32;
            let meta8 = transmute::<[u8; 8], u64>(self.child_meta);
            let is_inner8 = (meta8 & (meta8 << 1)) & 0x1010101010101010;
            let inner_mask8 = (is_inner8 >> 4) * 0xffu64;
            let bit_index8 = (meta8 ^ (oct_inv8 & inner_mask8)) & 0x1f1f1f1f1f1f1f1f;
            let child_bits8 = (meta8 >> 5) & 0x0707070707070707;

            #[inline(always)]
            fn get_q(v: &[u8; 8], i: usize) -> __m128 {
                // get_q is the most expensive part of intersect_simd
                // Tried version with _mm_cvtepu8_epi32 and _mm_cvtepi32_ps, it was a lot slower.
                unsafe {
                    _mm_set_ps(
                        *v.get_unchecked(i * 4 + 3) as f32,
                        *v.get_unchecked(i * 4 + 2) as f32,
                        *v.get_unchecked(i * 4 + 1) as f32,
                        *v.get_unchecked(i * 4) as f32,
                    )
                }
            }

            for i in 0..2 {
                let q_lo_x = get_q(&self.child_min_x, i);
                let q_lo_y = get_q(&self.child_min_y, i);
                let q_lo_z = get_q(&self.child_min_z, i);
                let q_hi_x = get_q(&self.child_max_x, i);
                let q_hi_y = get_q(&self.child_max_y, i);
                let q_hi_z = get_q(&self.child_max_z, i);

                let x_min = if rdx { q_hi_x } else { q_lo_x };
                let x_max = if rdx { q_lo_x } else { q_hi_x };
                let y_min = if rdy { q_hi_y } else { q_lo_y };
                let y_max = if rdy { q_lo_y } else { q_hi_y };
                let z_min = if rdz { q_hi_z } else { q_lo_z };
                let z_max = if rdz { q_lo_z } else { q_hi_z };

                // Compute tmin3 and tmax3
                // Tried using _mm_fmadd_ps, it was a lot slower
                let tmin_x = _mm_add_ps(_mm_mul_ps(x_min, adj_ray_dir_inv_x), adj_ray_orig_x);
                let tmax_x = _mm_add_ps(_mm_mul_ps(x_max, adj_ray_dir_inv_x), adj_ray_orig_x);
                let tmin_y = _mm_add_ps(_mm_mul_ps(y_min, adj_ray_dir_inv_y), adj_ray_orig_y);
                let tmax_y = _mm_add_ps(_mm_mul_ps(y_max, adj_ray_dir_inv_y), adj_ray_orig_y);
                let tmin_z = _mm_add_ps(_mm_mul_ps(z_min, adj_ray_dir_inv_z), adj_ray_orig_z);
                let tmax_z = _mm_add_ps(_mm_mul_ps(z_max, adj_ray_dir_inv_z), adj_ray_orig_z);

                let tmin = _mm_max_ps(tmin_x, _mm_max_ps(tmin_y, tmin_z));
                let tmax = _mm_min_ps(tmax_x, _mm_min_ps(tmax_y, tmax_z));

                // Compute intersection
                let tmin = _mm_max_ps(tmin, _mm_set1_ps(EPSILON)); //ray.tmin?
                let tmax = _mm_min_ps(tmax, _mm_set1_ps(ray.tmax));

                let intersected = _mm_cmple_ps(tmin, tmax);
                let mask = _mm_movemask_ps(intersected);

                for j in 0..4 {
                    let offset = i * 4 + j;
                    if (mask & (1 << j)) != 0 {
                        let child_bits = ((child_bits8 >> (offset * 8)) as u32) & 0xFF;
                        let bit_index = ((bit_index8 >> (offset * 8)) as u32) & 0xFF;

                        hit_mask |= child_bits << bit_index;
                    }
                }
            }
        }
        hit_mask
    }

    #[inline(always)]
    pub fn intersect_ray4_simd(&self, rays: &[Ray; 4], oct_inv4: u32) -> u32 {
        let extent = vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        );
        let p = Vec3A::from(self.p);

        let adj_ray_dir_inv = [
            extent * rays[0].inv_direction,
            extent * rays[1].inv_direction,
            extent * rays[2].inv_direction,
            extent * rays[3].inv_direction,
        ];
        let adj_ray_origins = [
            (p - rays[0].origin) * rays[0].inv_direction,
            (p - rays[1].origin) * rays[1].inv_direction,
            (p - rays[2].origin) * rays[2].inv_direction,
            (p - rays[3].origin) * rays[3].inv_direction,
        ];

        let mut hit_mask = 0;
        unsafe {
            let adj_ray_dir_inv_x = [
                _mm_set1_ps(adj_ray_dir_inv[0].x),
                _mm_set1_ps(adj_ray_dir_inv[1].x),
                _mm_set1_ps(adj_ray_dir_inv[2].x),
                _mm_set1_ps(adj_ray_dir_inv[3].x),
            ];
            let adj_ray_dir_inv_y = [
                _mm_set1_ps(adj_ray_dir_inv[0].y),
                _mm_set1_ps(adj_ray_dir_inv[1].y),
                _mm_set1_ps(adj_ray_dir_inv[2].y),
                _mm_set1_ps(adj_ray_dir_inv[3].y),
            ];
            let adj_ray_dir_inv_z = [
                _mm_set1_ps(adj_ray_dir_inv[0].z),
                _mm_set1_ps(adj_ray_dir_inv[1].z),
                _mm_set1_ps(adj_ray_dir_inv[2].z),
                _mm_set1_ps(adj_ray_dir_inv[3].z),
            ];

            let adj_ray_orig_x = [
                _mm_set1_ps(adj_ray_origins[0].x),
                _mm_set1_ps(adj_ray_origins[1].x),
                _mm_set1_ps(adj_ray_origins[2].x),
                _mm_set1_ps(adj_ray_origins[3].x),
            ];
            let adj_ray_orig_y = [
                _mm_set1_ps(adj_ray_origins[0].y),
                _mm_set1_ps(adj_ray_origins[1].y),
                _mm_set1_ps(adj_ray_origins[2].y),
                _mm_set1_ps(adj_ray_origins[3].y),
            ];
            let adj_ray_orig_z = [
                _mm_set1_ps(adj_ray_origins[0].z),
                _mm_set1_ps(adj_ray_origins[1].z),
                _mm_set1_ps(adj_ray_origins[2].z),
                _mm_set1_ps(adj_ray_origins[3].z),
            ];

            let mut oct_inv8 = oct_inv4 as u64;
            oct_inv8 |= oct_inv8 << 32;
            let meta8 = transmute::<[u8; 8], u64>(self.child_meta);
            let is_inner8 = (meta8 & (meta8 << 1)) & 0x1010101010101010;
            let inner_mask8 = (is_inner8 >> 4) * 0xffu64;
            let bit_index8 = (meta8 ^ (oct_inv8 & inner_mask8)) & 0x1f1f1f1f1f1f1f1f;
            let child_bits8 = (meta8 >> 5) & 0x0707070707070707;

            #[inline(always)]
            fn get_q(v: &[u8; 8], i: usize) -> __m128 {
                // get_q is the most expensive part of intersect_simd
                // Tried version with _mm_cvtepu8_epi32 and _mm_cvtepi32_ps, it was a lot slower.
                unsafe {
                    _mm_set_ps(
                        *v.get_unchecked(i * 4 + 3) as f32,
                        *v.get_unchecked(i * 4 + 2) as f32,
                        *v.get_unchecked(i * 4 + 1) as f32,
                        *v.get_unchecked(i * 4) as f32,
                    )
                }
            }

            for i in 0..2 {
                let q_lo_x = get_q(&self.child_min_x, i);
                let q_lo_y = get_q(&self.child_min_y, i);
                let q_lo_z = get_q(&self.child_min_z, i);
                let q_hi_x = get_q(&self.child_max_x, i);
                let q_hi_y = get_q(&self.child_max_y, i);
                let q_hi_z = get_q(&self.child_max_z, i);

                for n in 0..4 {
                    let rdx = rays[n].direction.x < 0.0;
                    let rdy = rays[n].direction.y < 0.0;
                    let rdz = rays[n].direction.z < 0.0;

                    let x_min = if rdx { q_hi_x } else { q_lo_x };
                    let x_max = if rdx { q_lo_x } else { q_hi_x };
                    let y_min = if rdy { q_hi_y } else { q_lo_y };
                    let y_max = if rdy { q_lo_y } else { q_hi_y };
                    let z_min = if rdz { q_hi_z } else { q_lo_z };
                    let z_max = if rdz { q_lo_z } else { q_hi_z };

                    // Compute tmin3 and tmax3
                    // Tried using _mm_fmadd_ps, it was a lot slower
                    let tmin_x =
                        _mm_add_ps(_mm_mul_ps(x_min, adj_ray_dir_inv_x[n]), adj_ray_orig_x[n]);
                    let tmax_x =
                        _mm_add_ps(_mm_mul_ps(x_max, adj_ray_dir_inv_x[n]), adj_ray_orig_x[n]);
                    let tmin_y =
                        _mm_add_ps(_mm_mul_ps(y_min, adj_ray_dir_inv_y[n]), adj_ray_orig_y[n]);
                    let tmax_y =
                        _mm_add_ps(_mm_mul_ps(y_max, adj_ray_dir_inv_y[n]), adj_ray_orig_y[n]);
                    let tmin_z =
                        _mm_add_ps(_mm_mul_ps(z_min, adj_ray_dir_inv_z[n]), adj_ray_orig_z[n]);
                    let tmax_z =
                        _mm_add_ps(_mm_mul_ps(z_max, adj_ray_dir_inv_z[n]), adj_ray_orig_z[n]);

                    let tmin = _mm_max_ps(tmin_x, _mm_max_ps(tmin_y, tmin_z));
                    let tmax = _mm_min_ps(tmax_x, _mm_min_ps(tmax_y, tmax_z));

                    // Compute intersection
                    let tmin = _mm_max_ps(tmin, _mm_set1_ps(EPSILON)); //ray.tmin?
                    let tmax = _mm_min_ps(tmax, _mm_set1_ps(rays[n].tmax));

                    let intersected = _mm_cmple_ps(tmin, tmax);
                    let mask = _mm_movemask_ps(intersected);

                    for j in 0..4 {
                        let offset = i * 4 + j;
                        if (mask & (1 << j)) != 0 {
                            let child_bits = ((child_bits8 >> (offset * 8)) as u32) & 0xFF;
                            let bit_index = ((bit_index8 >> (offset * 8)) as u32) & 0xFF;

                            hit_mask |= child_bits << bit_index;
                        }
                    }
                }
            }
        }
        hit_mask
    }
}
