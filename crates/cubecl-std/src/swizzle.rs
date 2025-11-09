use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Swizzling strategy for a buffer.
/// See the following docs from cutlass:
///
/// 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
///                                ^--^ MBase is the number of least-sig bits to keep constant
///                   ^-^       ^-^     BBits is the number of bits in the mask
///                     ^---------^     SShift is the distance to shift the YYY mask
///                                        (pos shifts YYY to the right, neg shifts YYY to the left)
///
/// # Example
/// Given:
/// 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
/// the result is:
/// 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct Swizzle {
    #[cube(comptime)]
    yyy_mask: u32,
    #[cube(comptime)]
    shift: u32,
    #[cube(comptime)]
    invert_shift: bool,
    /// Precalculate repeat after so we don't need to keep all the parts
    #[cube(comptime)]
    repeats_after: u32,
}

#[cube]
impl Swizzle {
    /// Create a new swizzle with comptime parameters
    pub fn new(#[comptime] bits: u32, #[comptime] base: u32, #[comptime] shift: i32) -> Self {
        let invert_shift = shift < 0;
        let mask = (1u32 << bits) - 1;
        let yyy_mask = comptime![mask << (base + Ord::max(shift, 0) as u32)];
        let repeats_after = comptime![1u32 << (base + bits + Ord::max(shift, 0) as u32)];
        Swizzle {
            yyy_mask,
            shift: comptime![shift.unsigned_abs()],
            invert_shift,
            repeats_after,
        }
    }

    /// Apply the swizzle to a coordinate with a given item size. This is the size of the full type,
    /// including line size. Use `type_size` helper for lines.
    /// `offset` should be in terms of lines from the start of the buffer, and the buffer should be
    /// aligned to `repeats_after`. This is to work around the fact we don't currently support
    /// retrieving the actual address of an offset.
    /// If you're using absolute/unlined indices, pass `E::type_size()` instead of the full line size.
    pub fn apply(&self, offset: u32, #[comptime] type_size: u32) -> u32 {
        let offset_bytes = offset * type_size;
        let offset_masked = offset_bytes & self.yyy_mask;
        let offset_shifted = shift_right(offset_masked, self.shift, comptime![self.invert_shift]);
        let offset_bytes = offset_bytes ^ offset_shifted;
        offset_bytes / type_size
    }

    /// After how many elements this pattern repeats. Can be used to align the buffer (i.e. smem)
    /// so offsets match addresses.
    pub fn repeats_after(&self) -> comptime_type!(u32) {
        self.repeats_after
    }
}

/// Retrieve the type size of a lined buffer.
#[cube]
pub fn type_size<E: CubePrimitive>(#[comptime] line_size: u32) -> comptime_type!(u32) {
    let storage_size = E::type_size();
    comptime![storage_size * line_size]
}

#[cube]
fn shift_right(value: u32, shift: u32, #[comptime] invert: bool) -> u32 {
    if invert {
        value << shift
    } else {
        value >> shift
    }
}
