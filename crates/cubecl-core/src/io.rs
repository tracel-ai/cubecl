use crate as cubecl;
use cubecl::prelude::*;

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, slice: Slice<C>, index: u32, value: C) -> C {
    let index = index * u32::cast_from(mask);
    let input = unsafe { slice.index_unchecked(index) };

    select(mask, *input, value)
}

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_checked<C: CubePrimitive>(slice: Slice<C>, index: u32) -> C {
    let mask = index < slice.len();

    read_masked::<C>(mask, slice, index, C::cast_from(0u32))
}
