// TODO re-enable needless_range_loop lint and evaluate performance / clarity
#![allow(clippy::needless_range_loop)]

//! # BVH Construction and Traversal Library
//!
//! - [PLOC](https://meistdan.github.io/publications/ploc/paper.pdf) BVH2 builder with [Parallel Reinsertion](https://meistdan.github.io/publications/prbvh/paper.pdf) and spatial pre-splits.
//! - [CWBVH](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf) An eight-way compressed wide BVH8 builder. Each BVH Node is compressed so that it takes up only 80 bytes per node.
//! - Tools for dynamically updating and optimizing the BVH2. ([Added in 0.3](https://github.com/DGriffin91/obvhs/pull/8))
//! - CPU traversal for both BVH2 and CWBVH (SIMD traversal, intersecting 4 nodes at a time)
//! - For GPU traversal example, see the [Tray Racing](https://github.com/DGriffin91/tray_racing) benchmark
//!
//! OBVHS optionally uses [rayon](https://github.com/rayon-rs/rayon) to parallelize building.
//!
//! ## Example
//!
//! ```
//! use glam::*;
//! use obvhs::{
//!     cwbvh::builder::build_cwbvh_from_tris,
//!     ray::{Ray, RayHit},
//!     test_util::geometry::{icosphere, PLANE},
//!     triangle::Triangle,
//!     BvhBuildParams,
//! };
//! use std::time::Duration;
//!
//!
//! // Build a scene with an icosphere and a plane
//! // BVH primitives do not need to be triangles, the BVH builder is only concerned with AABBs.
//! // (With the exception of optional precise triangle aabb splitting)
//! let mut tris: Vec<Triangle> = Vec::new();
//! tris.extend(icosphere(1));
//! tris.extend(PLANE);
//!
//! // Build the BVH.
//! // build_cwbvh_from_tris is just a helper that can build from BvhBuildParams and the
//! // respective presets. Feel free to copy the contents of build_cwbvh_from_tris or
//! // build_cwbvh. They are very straightforward. If you don't want to use Triangles as the
//! // primitive, use  build_cwbvh instead. build_cwbvh_from_tris just adds support for
//! // splitting tris.
//! let bvh = build_cwbvh_from_tris(
//!     &tris,
//!     BvhBuildParams::medium_build(),
//!     &mut Duration::default(),
//! );
//!
//! // Create a new ray
//! let ray = Ray::new_inf(vec3a(0.1, 0.1, 4.0), vec3a(0.0, 0.0, -1.0));
//!
//! // Traverse the BVH, finding the closest hit.
//! let mut ray_hit = RayHit::none();
//! if bvh.ray_traverse(ray, &mut ray_hit, |ray, id| {
//!     // Use primitive_indices to look up the original primitive id.
//!     // (Could reorder tris per bvh.primitive_indices to avoid this lookup, see
//!     // cornell_box_cwbvh example)
//!     tris[bvh.primitive_indices[id] as usize].intersect(ray)
//! }) {
//!     println!(
//!         "Hit Triangle {}",
//!         bvh.primitive_indices[ray_hit.primitive_id as usize]
//!     );
//!     println!("Distance to hit: {}", ray_hit.t);
//! } else {
//!     println!("Miss");
//! }
//!
//! ```

use std::time::Duration;

use aabb::Aabb;
use glam::Mat4;
use ploc::{PlocSearchDistance, SortPrecision};
use triangle::Triangle;

pub mod aabb;
pub mod bvh2;
pub mod cwbvh;
pub mod faststack;
pub mod ploc;
pub mod ray;
pub mod rt_triangle;
pub mod splits;
pub mod test_util;
pub mod triangle;

/// Used to indicate a vacant slot in various contexts.
#[doc(hidden)]
pub const INVALID: u32 = u32::MAX;

/// A trait for types that can be bounded by an axis-aligned bounding box (AABB). Used in Bvh2/CwBvh validation.
#[cfg(feature = "parallel")]
pub trait Boundable: Send + Sync {
    fn aabb(&self) -> Aabb;
}

/// A trait for types that can be bounded by an axis-aligned bounding box (AABB). Used in Bvh2/CwBvh validation.
#[cfg(not(feature = "parallel"))]
pub trait Boundable {
    fn aabb(&self) -> Aabb;
}

/// A trait for types that can have a matrix transform applied. Primarily for testing/examples.
pub trait Transformable {
    fn transform(&mut self, matrix: &Mat4);
}

/// Apply a function to each component of a type.
#[doc(hidden)]
pub trait PerComponent<C1, C2 = C1, Output = Self> {
    fn per_comp(self, f: impl Fn(C1) -> C2) -> Output;
}

impl<Input, C1, C2, Output> PerComponent<C1, C2, Output> for Input
where
    Input: Into<[C1; 3]>,
    Output: From<[C2; 3]>,
{
    /// Applies a function to each component of the input.
    fn per_comp(self, f: impl Fn(C1) -> C2) -> Output {
        let [x, y, z] = self.into();
        Output::from([f(x), f(y), f(z)])
    }
}

#[allow(unused)]
fn as_slice_of_atomic_u32(slice: &mut [u32]) -> &mut [core::sync::atomic::AtomicU32] {
    assert_eq!(size_of::<AtomicU32>(), size_of::<u32>());
    assert_eq!(align_of::<AtomicU32>(), align_of::<u32>());
    use core::sync::atomic::AtomicU32;
    let parents: &mut [AtomicU32] = unsafe {
        core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut AtomicU32, slice.len())
    };
    // Alternatively:
    //let slice: &mut [AtomicU32] = unsafe { &mut *((slice.as_mut_slice() as *mut [u32]) as *mut [AtomicU32]) };
    parents
}

/// A macro to measure and print the execution time of a block of code.
///
/// # Arguments
/// * `$label` - A string label to identify the code block being timed.
/// * `$($code:tt)*` - The code block whose execution time is to be measured.
///
/// # Usage
/// ```rust
/// use obvhs::timeit;
/// timeit!["example",
///     // code to measure
/// ];
/// ```
///
/// # Note
/// The macro purposefully doesn't include a scope so variables don't need to
/// be passed out of it. This allows it to be trivially added to existing code.
///
/// This macro only measures time when the `timeit` feature is enabled.
#[macro_export]
#[doc(hidden)]
macro_rules! timeit {
    [$label:expr, $($code:tt)*] => {
        #[cfg(feature = "timeit")]
        let timeit_start = std::time::Instant::now();
        $($code)*
        #[cfg(feature = "timeit")]
        println!("{:>8} {}", format!("{}", $crate::PrettyDuration(timeit_start.elapsed())), $label);
    };
}

/// A wrapper struct for `std::time::Duration` to provide pretty-printing of durations.
#[doc(hidden)]
pub struct PrettyDuration(pub Duration);

impl std::fmt::Display for PrettyDuration {
    /// Durations are formatted as follows:
    /// - If the duration is greater than or equal to 1 second, it is formatted in seconds (s).
    /// - If the duration is greater than or equal to 1 millisecond but less than 1 second, it is formatted in milliseconds (ms).
    /// - If the duration is less than 1 millisecond, it is formatted in microseconds (µs).
    ///   In the case of seconds & milliseconds, the duration is always printed with a precision of two decimal places.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = self.0;
        if duration.as_secs() > 0 {
            let seconds =
                duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;
            write!(f, "{seconds:.2}s ")
        } else if duration.subsec_millis() > 0 {
            let milliseconds =
                duration.as_millis() as f64 + f64::from(duration.subsec_micros() % 1_000) / 1_000.0;
            write!(f, "{milliseconds:.2}ms")
        } else {
            let microseconds = duration.as_micros();
            write!(f, "{microseconds}µs")
        }
    }
}

/// Add profile scope. Nesting the macro allows us to make the profiling crate optional.
#[doc(hidden)]
#[macro_export]
macro_rules! scope {
    [$label:expr] => {
        #[cfg(feature = "profile")]
        profiling::scope!($label);
    };
}

/// General build parameters for Bvh2 & CwBvh
#[derive(Clone, Copy, Debug)]
pub struct BvhBuildParams {
    /// Split large tris into multiple AABBs
    pub pre_split: bool,
    /// In ploc, the number of nodes before and after the current one that are evaluated for pairing. 1 has a
    /// fast path in building and still results in decent quality BVHs esp. when paired with a bit of reinsertion.
    pub ploc_search_distance: PlocSearchDistance,
    /// Below this depth a search distance of 1 will be used for ploc.
    pub search_depth_threshold: usize,
    /// Typically 0..1: ratio of nodes considered as candidates for reinsertion. Above 1 to evaluate the whole set
    /// multiple times. A little goes a long way. Try 0.01 or even 0.001 before disabling for build performance.
    pub reinsertion_batch_ratio: f32,
    /// For Bvh2 only, a second pass of reinsertion after collapse. Since collapse reduces the node count,
    /// this reinsertion pass will be faster. 0 to disable. Relative to the initial reinsertion_batch_ratio.
    pub post_collapse_reinsertion_batch_ratio_multiplier: f32,
    /// Bits used for ploc radix sort.
    pub sort_precision: SortPrecision,
    /// Min 1 (CwBvh will clamp to max 3, Bvh2 will clamp to max 255)
    pub max_prims_per_leaf: u32,
    /// Multiplier for traversal cost calculation during Bvh2 collapse (Does not affect CwBvh). A higher value will
    /// result in more primitives per leaf.
    pub collapse_traversal_cost: f32,
}

impl BvhBuildParams {
    pub const fn fastest_build() -> Self {
        BvhBuildParams {
            pre_split: false,
            ploc_search_distance: PlocSearchDistance::Minimum,
            search_depth_threshold: 0,
            reinsertion_batch_ratio: 0.0,
            post_collapse_reinsertion_batch_ratio_multiplier: 0.0,
            sort_precision: SortPrecision::U64,
            max_prims_per_leaf: 1,
            collapse_traversal_cost: 1.0,
        }
    }
    pub const fn very_fast_build() -> Self {
        BvhBuildParams {
            pre_split: false,
            ploc_search_distance: PlocSearchDistance::Minimum,
            search_depth_threshold: 0,
            reinsertion_batch_ratio: 0.01,
            post_collapse_reinsertion_batch_ratio_multiplier: 0.0,
            sort_precision: SortPrecision::U64,
            max_prims_per_leaf: 8,
            collapse_traversal_cost: 3.0,
        }
    }
    pub const fn fast_build() -> Self {
        BvhBuildParams {
            pre_split: false,
            ploc_search_distance: PlocSearchDistance::Low,
            search_depth_threshold: 2,
            reinsertion_batch_ratio: 0.02,
            post_collapse_reinsertion_batch_ratio_multiplier: 0.0,
            sort_precision: SortPrecision::U64,
            max_prims_per_leaf: 8,
            collapse_traversal_cost: 3.0,
        }
    }
    /// Tries to be around the same build time as embree but with faster traversal
    pub const fn medium_build() -> Self {
        BvhBuildParams {
            pre_split: false,
            ploc_search_distance: PlocSearchDistance::Medium,
            search_depth_threshold: 3,
            reinsertion_batch_ratio: 0.05,
            post_collapse_reinsertion_batch_ratio_multiplier: 2.0,
            sort_precision: SortPrecision::U64,
            max_prims_per_leaf: 8,
            collapse_traversal_cost: 3.0,
        }
    }
    pub const fn slow_build() -> Self {
        BvhBuildParams {
            pre_split: true,
            ploc_search_distance: PlocSearchDistance::High,
            search_depth_threshold: 2,
            reinsertion_batch_ratio: 0.2,
            post_collapse_reinsertion_batch_ratio_multiplier: 2.0,
            sort_precision: SortPrecision::U128,
            max_prims_per_leaf: 8,
            collapse_traversal_cost: 3.0,
        }
    }
    pub const fn very_slow_build() -> Self {
        BvhBuildParams {
            pre_split: true,
            ploc_search_distance: PlocSearchDistance::Medium,
            search_depth_threshold: 1,
            reinsertion_batch_ratio: 1.0,
            post_collapse_reinsertion_batch_ratio_multiplier: 1.0,
            sort_precision: SortPrecision::U128,
            max_prims_per_leaf: 8,
            collapse_traversal_cost: 3.0,
        }
    }
}
