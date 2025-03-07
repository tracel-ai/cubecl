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
pub trait RadixData:
    CubeType<ExpandType = ExpandElementTyped<Self>> + Sized + CubePrimitive
{
    const BYTECNT: u32;
    fn shift_mask(this: &Self, shift: u32, mask: u32) -> u32;
}

#[cube]
impl RadixData for u32 {
    const BYTECNT: u32 = 4;
    fn shift_mask(this: &Self, shift: u32, mask: u32) -> u32 {
        (this >> shift) & mask
    }
}
