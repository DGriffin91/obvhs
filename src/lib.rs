use std::time::Duration;

use aabb::Aabb;
use glam::Mat4;
use ploc::{PlocSearchDistance, SortPrecision};
use triangle::Triangle;

pub mod aabb;
pub mod bvh2;
pub mod cwbvh;
pub mod heapstack;
pub mod ploc;
pub mod ray;
pub mod rt_triangle;
pub mod splits;
pub mod test_util;
pub mod triangle;

/// A trait for types that can be bounded by an axis-aligned bounding box (AABB). Used in Bvh2/CwBvh validation.
pub trait Boundable {
    fn aabb(&self) -> Aabb;
}

/// A trait for types that can have a matrix transform applied. Primarily for testing/examples.
pub trait Transformable {
    fn transform(&mut self, matrix: &Mat4);
}

/// Apply a function to each component of a type.
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

#[doc(hidden)]
pub trait VecExt {
    /// Computes the base 2 logarithm of each component of the vector.
    fn log2(self) -> Self;
    /// Computes the base 2 exponential of each component of the vector.
    fn exp2(self) -> Self;
}

impl VecExt for glam::Vec3 {
    /// Computes the base 2 logarithm of each component of the `Vec3` vector.
    fn log2(self) -> Self {
        self.per_comp(f32::log2)
    }

    /// Computes the base 2 exponential of each component of the `Vec3` vector.
    fn exp2(self) -> Self {
        self.per_comp(f32::exp2)
    }
}

impl VecExt for glam::Vec3A {
    /// Computes the base 2 logarithm of each component of the `Vec3A` vector.
    fn log2(self) -> Self {
        self.per_comp(f32::log2)
    }

    /// Computes the base 2 exponential of each component of the `Vec3A` vector.
    fn exp2(self) -> Self {
        self.per_comp(f32::exp2)
    }
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
    /// In the case of seconds & milliseconds, the duration is always printed with a precision of two decimal places.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = self.0;
        if duration.as_secs() > 0 {
            let seconds =
                duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;
            write!(f, "{:.2}s ", seconds)
        } else if duration.subsec_millis() > 0 {
            let milliseconds =
                duration.as_millis() as f64 + f64::from(duration.subsec_micros() % 1_000) / 1_000.0;
            write!(f, "{:.2}ms", milliseconds)
        } else {
            let microseconds = duration.as_micros();
            write!(f, "{}µs", microseconds)
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
    /// Min 1 (CwBvh will clamp to max 3)
    pub max_prims_per_leaf: u32,
    /// Multiplier for traversal cost calculation during Bvh2 collapse (Does not affect CwBvh). A higher value will
    /// result in more primitives per leaf.
    pub collapse_traversal_cost: f32,
}

impl BvhBuildParams {
    pub fn fastest_build() -> Self {
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
    pub fn very_fast_build() -> Self {
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
    pub fn fast_build() -> Self {
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
    pub fn medium_build() -> Self {
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
    pub fn slow_build() -> Self {
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
    pub fn very_slow_build() -> Self {
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
