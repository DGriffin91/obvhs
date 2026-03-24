use std::{mem, time::Instant};

use obvhs::sort::{output_will_be_in_input, radix_sort};

pub fn uhash(x: u32) -> u32 {
    // from https://nullprogram.com/blog/2018/07/31/
    let mut x = x ^ (x >> 16);
    x = x.overflowing_mul(0x7feb352d).0;
    x = x ^ (x >> 15);
    x = x.overflowing_mul(0x846ca68b).0;
    x = x ^ (x >> 16);
    x
}

#[inline(always)]
pub fn hash64(x: u64) -> u64 {
    // https://nullprogram.com/blog/2018/07/31/
    let mut x = x ^ (x >> 30);
    x = x.overflowing_mul(0xbf58476d1ce4e5b9).0;
    x = x ^ (x >> 27);
    x = x.overflowing_mul(0x94d049bb133111eb).0;
    x = x ^ (x >> 31);
    x
}

fn main() {
    let qty = 10_000_000;

    // warm
    let mut a = (0..qty).map(|n| (hash64(n), 0u64)).collect::<Vec<_>>();
    let mut b = vec![(0, 0); a.len()];
    radix_sort::<1024, _>(&mut a, &mut b, 64);
    if !output_will_be_in_input::<1024>(64) {
        mem::swap(&mut a, &mut b);
    }
    assert!(a.is_sorted());

    let mut sum = 0.0;
    let n = 4;
    for _ in 0..n {
        let mut a = (0..qty).map(|n| (hash64(n), 0u64)).collect::<Vec<_>>();
        let start = Instant::now();
        let mut b = vec![(0, 0); a.len()];
        let maxv = u64::MAX as usize;
        let max_bits = maxv.ilog2() + 1;
        radix_sort::<1024, _>(&mut a, &mut b, max_bits);
        if !output_will_be_in_input::<1024>(max_bits) {
            mem::swap(&mut a, &mut b);
        }
        sum += start.elapsed().as_secs_f64();
        assert!(a.is_sorted());
    }
    println!("{:.5}ms\tradix sort small64", sum / (n as f64) * 1000.0);

    let mut sum = 0.0;
    for _ in 0..n {
        let mut a = (0..qty).map(|n| (hash64(n), 0u64)).collect::<Vec<_>>();
        let start = Instant::now();
        a.sort_unstable_by_key(|(k, _v)| *k);
        sum += start.elapsed().as_secs_f64();
        assert!(a.is_sorted());
    }
    println!("{:.5}ms\tsort_unstable_by_key", sum / (n as f64) * 1000.0);
}
