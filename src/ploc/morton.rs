// http://www.graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
// TODO evaluate Extended Morton Codes for High Performance Bounding Volume Hierarchy Construction:
// https://www.dcgi.fel.cvut.cz/projects/emc/emc2017.pdf
// https://www.highperformancegraphics.org/wp-content/uploads/2017/Papers-Session3/HPG207_ExtendedMortonCodes.pdf

//---------------------------------------------------
// --- 10 bit resolution per channel morton curve ---
//---------------------------------------------------

use glam::DVec3;

#[inline]
pub fn split_by_3_u32(a: u16) -> u32 {
    let mut x = a as u32 & 0x3ff; // we only look at the first 10 bits
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    x
}

#[inline]
/// Encode x,y,z position into a u64 morton value.
/// Input should be 0..=2u16.pow(10) (or 1u16 << 10)
/// (only included for reference, this isn't reasonably accurate for most BVHs)
pub fn morton_encode_u32(x: u16, y: u16, z: u16) -> u32 {
    split_by_3_u32(x) | split_by_3_u32(y) << 1 | split_by_3_u32(z) << 2
}

//---------------------------------------------------
// --- 21 bit resolution per channel morton curve ---
//---------------------------------------------------

#[inline]
pub fn split_by_3_u64(a: u32) -> u64 {
    let mut x = a as u64 & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    x
}

#[inline]
/// Encode x,y,z position into a u64 morton value.
/// Input should be 0..=2u32.pow(21) (or 1u32 << 21)
pub fn morton_encode_u64(x: u32, y: u32, z: u32) -> u64 {
    split_by_3_u64(x) | split_by_3_u64(y) << 1 | split_by_3_u64(z) << 2
}

#[inline]
/// Encode a DVec3 position into a u128 morton value.
/// Input should be 0.0..=1.0
pub fn morton_encode_u64_unorm(p: DVec3) -> u64 {
    let p = p * (1 << 21) as f64;
    morton_encode_u64(p.x as u32, p.y as u32, p.z as u32)
}

//---------------------------------------------------
// --- 42 bit resolution per channel morton curve ---
//---------------------------------------------------

#[inline]
pub fn split_by_3_u128(a: u64) -> u128 {
    let mut x = a as u128 & 0x3ffffffffff; // we only look at the first 42 bits
    x = (x | x << 64) & 0x3ff0000000000000000ffffffff;
    x = (x | x << 32) & 0x3ff00000000ffff00000000ffff;
    x = (x | x << 16) & 0x30000ff0000ff0000ff0000ff0000ff;
    x = (x | x << 8) & 0x300f00f00f00f00f00f00f00f00f00f;
    x = (x | x << 4) & 0x30c30c30c30c30c30c30c30c30c30c3;
    x = (x | x << 2) & 0x9249249249249249249249249249249;
    x
}

#[inline]
/// Encode x,y,z position into a u128 morton value.
/// Input should be 0..=2u64.pow(42) (or 1u64 << 42)
pub fn morton_encode_u128(x: u64, y: u64, z: u64) -> u128 {
    split_by_3_u128(x) | split_by_3_u128(y) << 1 | split_by_3_u128(z) << 2
}

#[inline]
/// Encode a DVec3 position into a u128 morton value.
/// Input should be 0.0..=1.0
pub fn morton_encode_u128_unorm(p: DVec3) -> u128 {
    let p = p * (1u64 << 42) as f64;
    morton_encode_u128(p.x as u64, p.y as u64, p.z as u64)
}
