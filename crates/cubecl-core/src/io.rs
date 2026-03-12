use crate as cubecl;
use cubecl::prelude::*;

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, list: Slice<C>, index: usize, value: C) -> C {
    let index = index * usize::cast_from(mask);
    let input = list.read_unchecked(index);

    select(mask, input, value)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_checked<C: CubePrimitive + Default + IntoRuntime>(
    tensor: Tensor<C>,
    index: usize,
    #[comptime] unroll_factor: usize,
) -> C {
    let len = tensor.buffer_len() * unroll_factor;
    let in_bounds = index < len;
    let index = index.min(len);

    select(in_bounds, tensor.read_unchecked(index), C::default())
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_atomic_checked<C: Scalar>(
    tensor: Tensor<Atomic<C>>,
    index: usize,
    #[comptime] unroll_factor: usize,
) -> Atomic<C> {
    let index = index.min(tensor.buffer_len() * unroll_factor);

    tensor.read_unchecked(index)
}
