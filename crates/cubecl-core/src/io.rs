use crate as cubecl;
use cubecl::prelude::*;

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, list: Slice<C>, index: u32, value: C) -> C {
    comptime! {
                println!("read_masked");
    };
    let index = index * u32::cast_from(mask);
    let input = list.read_unchecked(index);

    select(mask, input, value)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_checked<C: CubePrimitive>(tensor: Array<C>, index: u32) -> C {
    comptime! {
                println!("read tensor checked");
    };
    let mask = index < tensor.buffer_len();

    read_masked::<C>(mask, tensor.to_slice(), index, C::cast_from(0u32))
}
