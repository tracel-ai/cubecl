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
///
///
/// Some newer features, as well as cutlass in places, use a different terminology of `span` and
/// `atom`. For shared memory swizzle specifically, the parameters map as follows:
/// * `bits` = `log2(span / atom)`, or the number of atoms within one span, converted to address bits
/// * `base` = `log2(atom)`, the size of the atom, converted to address bits
/// * `shift` = `log2(all_banks_bytes / atom)`, or the total number of atoms in all 32 shared memory banks, converted to address bits
///
/// For example:
/// * 32-byte span with a 16-byte atom = `[1, 4, 3]`
/// * 128-byte span with a 32-byte atom = `[3, 5, 2]`
///
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
        let repeats_after = comptime![if bits > 0 {
            1u32 << (base + bits + Ord::max(shift, 0) as u32)
        } else {
            1u32 << base
        }];
        Swizzle {
            yyy_mask,
            shift: comptime![shift.unsigned_abs()],
            invert_shift,
            repeats_after,
        }
    }

    /// Create a new noop swizzle object
    pub fn none() -> Self {
        Swizzle {
            yyy_mask: 0u32,
            shift: 0u32,
            invert_shift: false,
            repeats_after: 1u32,
        }
    }

    /// Apply the swizzle to a coordinate with a given item size. This is the size of the full type,
    /// including line size. Use `type_size` helper for lines.
    /// `offset` should be in terms of lines from the start of the buffer, and the buffer should be
    /// aligned to `repeats_after`. This is to work around the fact we don't currently support
    /// retrieving the actual address of an offset.
    /// If you're using absolute/unlined indices, pass `E::type_size()` instead of the full line size.
    pub fn apply(&self, offset: u32, #[comptime] type_size: u32) -> u32 {
        // Special case here so we don't need to special case in kernels that can have no swizzle.
        // If `yyy_mask == 0`, the whole thing is a noop.
        if comptime![self.yyy_mask == 0] {
            offset
        } else {
            let offset_bytes = offset * type_size;
            let offset_masked = offset_bytes & self.yyy_mask;
            let offset_shifted =
                shift_right(offset_masked, self.shift, comptime![self.invert_shift]);
            let offset_bytes = offset_bytes ^ offset_shifted;
            offset_bytes / type_size
        }
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
