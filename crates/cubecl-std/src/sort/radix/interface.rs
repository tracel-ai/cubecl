use cubecl_core as cubecl;

use cubecl::prelude::*;

#[cube]
pub trait AsRadix: CubePrimitive + Numeric {
    type RadixType: RadixData;
    fn as_radix(this: &Self) -> Self::RadixType;
}

#[cube]
impl AsRadix for u32 {
    type RadixType = u32;
    fn as_radix(this: &Self) -> Self::RadixType {
        *this
    }
}

#[cube]
impl AsRadix for i32 {
    type RadixType = u32;
    fn as_radix(this: &Self) -> Self::RadixType {
        u32::bitcast_from(*this) ^ 0x80_00_00_00u32
    }
}

#[cube]
impl AsRadix for f32 {
    type RadixType = u32;
    fn as_radix(this: &Self) -> Self::RadixType {
        let bits = u32::bitcast_from(*this);
        if (bits & 0x80_00_00_00u32) != 0 {
            BitwiseNot::bitwise_not(bits)
        } else {
            bits | 0x80_00_00_00u32
        }
    }
}

#[cube]
pub trait RadixData:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + Sized
    + CubePrimitive
{
    const BYTECNT: u32;
    // const MAX: Self;
    fn shift_mask(
        this: &Self,
        shift: u32,
        mask: u32,
    ) -> u32;
}

#[cube]
impl RadixData for u32 {
    const BYTECNT: u32 = 4;
    // const MAX: Self = u32::MAX;
    fn shift_mask(
        this: &Self,
        shift: u32,
        mask: u32,
    ) -> u32 {
        // T::max_value();
        // u32::cast_from(*this >> T::cast_from(shift)) & mask
        this >> shift & mask
    }
}

// #[cube]
// impl<T: Int> RadixData for T {
//     const BYTECNT: u32 = T::BITS / 8;
//     fn shift_mask(
//         this: &Self,
//         shift: u32,
//         mask: u32,
//     ) -> u32 {
//         T::max_value();
//         u32::cast_from(*this >> T::cast_from(shift)) & mask
//     }
// }
