use crate as cubecl;
use cubecl::prelude::*;
use cubecl_ir::pliron::{common_traits::Named, value::Value};

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, list: &[C], index: usize, value: C) -> C {
    let index = index * usize::cast_from(mask);
    let input = unsafe { *list.get_unchecked(index) };

    select(mask, input, value)
}

/// Returns the value at `index` in `list` if the index is in bounds, otherwise returns `value`.
#[cube]
pub fn read_checked<C: CubePrimitive>(list: &[C], index: usize) -> C {
    let fallback = comptime![C::Scalar::default()].runtime();
    let clamped = index.min(list.len() - 1);
    let input = unsafe { *list.get_unchecked(clamped) };

    select(index == clamped, input, C::cast_from(fallback))
}

/// Writes the value only if it is in bounds of the buffer
#[cube]
pub fn write_checked<C: CubePrimitive>(list: &mut [C], index: usize, value: C) {
    if index < list.len() {
        unsafe { *list.get_unchecked_mut(index) = value };
    }
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn checked_index(index: usize, buffer_len: usize, #[comptime] unroll_factor: usize) -> usize {
    let len = buffer_len * unroll_factor;
    index.min(len - 1)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn validate_index(
    #[comptime] buffer_name: &str,
    index: usize,
    buffer_len: usize,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: &str,
) -> usize {
    let len = buffer_len * unroll_factor;
    let in_bounds = index < len;
    if !in_bounds {
        print_oob(kernel_name, index, len, buffer_name);
    }

    index.min(len)
}

#[cube]
#[allow(unused)]
fn print_oob(
    #[comptime] kernel_name: &str,
    index: usize,
    len: usize,
    #[comptime] buffer_name: &str,
) {
    intrinsic!(|scope| {
        __expand_debug_print!(
            scope,
            alloc::format!(
                "[VALIDATION {kernel_name}]: Encountered OOB index in {buffer_name} at %u, length is %u\n"
            ),
            index,
            len
        );
    })
}

#[allow(missing_docs)]
pub fn expand_checked_index(
    scope: &Scope,
    list: Value,
    index: Value,
    unroll_factor: usize,
) -> Value {
    let len = expand_buffer_length_native(scope, list);
    let index = checked_index::expand(scope, index.into(), len.into(), unroll_factor);
    index_expand(scope, list, index.value(scope), false)
}

#[allow(missing_docs)]
pub fn expand_validate_index(
    scope: &Scope,
    list: Value,
    index: Value,
    unroll_factor: usize,
    kernel_name: &str,
) -> Value {
    let len = expand_buffer_length_native(scope, list);
    let buffer_name = list.given_name(&scope.ctx());
    let buffer_name = buffer_name
        .as_ref()
        .map(|it| it.as_str())
        .unwrap_or("buffer");
    let index = validate_index::expand(
        scope,
        buffer_name,
        index.into(),
        len.into(),
        unroll_factor,
        kernel_name,
    );
    index_expand(scope, list, index.value(scope), false)
}
