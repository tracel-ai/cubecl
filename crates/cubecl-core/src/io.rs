use crate as cubecl;
use cubecl::prelude::*;

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, list: Slice<C>, index: u32, value: C) -> C {
    let index = index * u32::cast_from(mask);
    let input = list.read_unchecked(index);

    select(mask, input, value)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_checked<C: CubePrimitive>(
    tensor: Tensor<C>,
    index: u32,
    #[comptime] unroll_factor: u32,
) -> C {
    let len = tensor.buffer_len() * unroll_factor;
    let in_bounds = index < len;
    let index = index.min(len);

    select(in_bounds, tensor.read_unchecked(index), C::cast_from(0u32))
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_atomic_checked<C: Numeric>(
    tensor: Tensor<Atomic<Line<C>>>,
    index: u32,
    #[comptime] unroll_factor: u32,
) -> Atomic<Line<C>> {
    let index = index.min(tensor.buffer_len() * unroll_factor);

    tensor.read_unchecked(index)
}
