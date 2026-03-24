pub trait RadixBin {
    fn bin(&self, shift: usize, mask: usize) -> u8;
}

impl RadixBin for (u64, u64) {
    #[inline(always)]
    fn bin(&self, shift: usize, mask: usize) -> u8 {
        ((self.0 >> shift as u64) & mask as u64) as u8
    }
}

// To calculate max_bits: `max_key_value.ilog2() + 1`
// Works on arrays with up to u32::MAX items
// Taking a &mut Vec<T> lets us swap the original pointers so the result is always in `input`. We could take &mut [T]
// but then to avoid copying we would need to indicate if input or temp has the result
pub fn radix_sort<T: RadixBin + Copy>(input: &mut Vec<T>, temp: &mut Vec<T>, max_bits: u32) {
    const BINS: usize = 256;
    assert!(BINS.is_power_of_two());
    assert_eq!(input.len(), temp.len());
    assert!(
        input.len() <= u32::MAX as usize,
        "Sorting more than u32::MAX items is not supported."
    );
    const PASS_SHIFT: usize = BINS.ilog2() as usize;
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

        std::mem::swap(input, temp);
        bins.fill(0); // reset bins
        shift += PASS_SHIFT;
    }
}

#[cfg(test)]
mod tests {
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
            for div in [1, 3, 7, 13] {
                let mut a = (0..qty)
                    .map(|n| (uhash(n) as u64 / div, 0u64))
                    .collect::<Vec<_>>();
                let mut output = vec![(0, 0); a.len()];
                let maxv = u32::MAX as usize / div as usize;
                let max_bits = maxv.ilog2() + 1;
                radix_sort(&mut a, &mut output, max_bits);
                assert!(a.is_sorted());
            }
        }
    }

    #[test]
    fn large_arrays() {
        // works on arrays with up to u32::MAX items, but don't want test to be too slow.
        let qty = 10_000_000;
        let mut a = (0..qty).map(|n| (uhash(n) as u64, 0)).collect::<Vec<_>>();
        let mut output = vec![(0, 0); a.len()];
        radix_sort(&mut a, &mut output, u32::MAX.ilog2() + 1);
        assert!(a.is_sorted());
    }

    #[test]
    fn identical_items() {
        let mut a = vec![(100, 0); 1000];
        let mut output = vec![(0, 0); a.len()];
        radix_sort(&mut a, &mut output, u32::MAX.ilog2() + 1);
        assert!(a.is_sorted());
        a.push((1, 0));
        a.push((5, 0));
        output.push((0, 0));
        output.push((0, 0));
        radix_sort(&mut a, &mut output, u32::MAX.ilog2() + 1);
        assert!(a.is_sorted());
    }
}
