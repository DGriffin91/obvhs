pub trait RadixBin {
    fn bin(&self, shift: usize, mask: usize) -> usize;
}

impl RadixBin for (u64, u64) {
    #[inline(always)]
    fn bin(&self, shift: usize, mask: usize) -> usize {
        ((self.0 as usize >> shift) & mask) as usize
    }
}

pub fn output_will_be_in_input<const BINS: usize>(max_bits: u32) -> bool {
    let pass_shift: usize = BINS.ilog2() as usize;
    max_bits.div_ceil(pass_shift as u32) % 2 == 0
}

// To calculate max_bits: `max_key_value.ilog2() + 1`
// Works on arrays with up to u32::MAX items
// if returns false the output is in temp rather than in input
pub fn radix_sort<const BINS: usize, T: RadixBin + Copy>(
    input: &mut [T],
    temp: &mut [T],
    max_bits: u32,
) {
    assert!(BINS.is_power_of_two());
    assert_eq!(input.len(), temp.len());
    assert!(
        input.len() <= u32::MAX as usize,
        "Sorting more than u32::MAX items is not supported."
    );
    let pass_shift: usize = BINS.ilog2() as usize;
    let mut input = input;
    let mut temp = temp;

    let mask = BINS - 1;
    let mut bins = [0u32; BINS];
    let mut shift = 0usize;
    while shift < max_bits as usize {
        // count up items per bin
        for i in 0..input.len() {
            bins[input[i].bin(shift, mask) as usize] += 1;
        }

        // Change bins[i] so that bins[i] now contains actual position of this digit in temp
        for bin in 1..BINS {
            bins[bin] += bins[bin - 1];
        }

        // Build the temp array
        for i in (0..input.len()).rev() {
            let v = input[i];
            let bin = v.bin(shift, mask) as usize;
            let count = bins[bin] - 1;
            bins[bin] = count;
            temp[count as usize] = v;
        }

        std::mem::swap(&mut input, &mut temp);
        bins.fill(0); // reset bins
        shift += pass_shift;
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use super::*;

    pub fn uhash(x: u32) -> u32 {
        // from https://nullprogram.com/blog/2018/07/31/
        let mut x = x ^ (x >> 16);
        x = x.overflowing_mul(0x7feb352d).0;
        x = x ^ (x >> 15);
        x = x.overflowing_mul(0x846ca68b).0;
        x = x ^ (x >> 16);
        x
    }

    #[test]
    fn test_variations() {
        // verify sorting works on many different sized arrays
        for qty in [0, 1, 2, 3, 4, 5, 11, 78, 643, 1234, 85936, 213424, 3245343] {
            // verify sorting works on many different numbers of bits
            for div in [1, 1234, 1234567] {
                let mut a = (0..qty)
                    .map(|n| (uhash(n) as u64 / div, 0u64))
                    .collect::<Vec<_>>();
                let mut b = vec![(0, 0); a.len()];
                let maxv = u32::MAX as usize / div as usize;
                let max_bits = maxv.ilog2() + 1;
                radix_sort::<256, _>(&mut a, &mut b, max_bits);
                if !output_will_be_in_input::<256>(max_bits) {
                    mem::swap(&mut a, &mut b);
                }
                assert!(a.is_sorted());
            }
        }
    }

    #[test]
    fn large_arrays() {
        // works on arrays with up to u32::MAX items, but don't want test to be too slow.
        let qty = 10_000_000;
        let mut a = (0..qty).map(|n| (uhash(n) as u64, 0)).collect::<Vec<_>>();
        let mut b = vec![(0, 0); a.len()];
        let max_bits = u32::MAX.ilog2() + 1;
        radix_sort::<256, _>(&mut a, &mut b, max_bits);
        if !output_will_be_in_input::<256>(max_bits) {
            mem::swap(&mut a, &mut b);
        }
        assert!(a.is_sorted());
    }

    #[test]
    fn identical_items() {
        let mut a = vec![(100, 0); 1000];
        let mut b = vec![(0, 0); a.len()];
        let max_bits = u32::MAX.ilog2() + 1;
        radix_sort::<256, _>(&mut a, &mut b, max_bits);
        if !output_will_be_in_input::<256>(max_bits) {
            mem::swap(&mut a, &mut b);
        }
        assert!(a.is_sorted());
        a.push((1, 0));
        a.push((5, 0));
        b.push((0, 0));
        b.push((0, 0));
        radix_sort::<256, _>(&mut a, &mut b, max_bits);
        if !output_will_be_in_input::<256>(max_bits) {
            mem::swap(&mut a, &mut b);
        }
        assert!(a.is_sorted());
    }
}
