use alloc::{
    borrow::Cow,
    string::{String, ToString},
};

use crate as cubecl;
use cubecl::prelude::*;
use cubecl_ir::{Instruction, Operation, Variable};

define_scalar!(ElemA);
define_size!(SizeA);

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
pub fn checked_index<E: Scalar, N: Size>(
    index: usize,
    buffer_len: usize,
    #[comptime] unroll_factor: usize,
) -> usize {
    let len = buffer_len * unroll_factor;
    index.min(len - 1)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn validate_index<E: Scalar, N: Size>(
    tensor: &[Vector<E, N>],
    index: usize,
    buffer_len: usize,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) -> usize {
    let len = buffer_len * unroll_factor;
    let in_bounds = index < len;
    if !in_bounds {
        print_oob::<[Vector<E, N>]>(kernel_name, index, len, tensor);
    }

    index.min(len)
}

#[cube]
#[allow(unused)]
fn print_oob<Out: CubeType<ExpandType: Into<Variable>> + ?Sized>(
    #[comptime] kernel_name: String,
    index: usize,
    len: usize,
    buffer: &Out,
) {
    intrinsic!(|scope| {
        let name = name_of_var(scope, buffer.clone_unchecked().into());
        __expand_debug_print!(
            scope,
            alloc::format!(
                "[VALIDATION {kernel_name}]: Encountered OOB index in {name} at %u, length is %u\n"
            ),
            index,
            len
        );
    })
}

fn name_of_var(scope: &Scope, var: Variable) -> Cow<'static, str> {
    let debug_name = scope.debug.variable_names.borrow().get(&var).cloned();
    debug_name.unwrap_or_else(|| var.to_string().into())
}

#[allow(missing_docs)]
pub fn expand_checked_index(
    scope: &Scope,
    list: Variable,
    index: Variable,
    out: Variable,
    unroll_factor: usize,
) {
    scope.register_type::<ElemA>(list.ty.storage_type());
    scope.register_size::<SizeA>(list.ty.vector_size());
    let len = expand_buffer_length_native(scope, list);
    let index =
        checked_index::expand::<ElemA, SizeA>(scope, index.into(), len.into(), unroll_factor);
    let ptr = index_expand(scope, list, index.expand, None, false);
    scope.register(Instruction::new(Operation::Copy(ptr), out));
}

#[allow(missing_docs)]
pub fn expand_validate_index(
    scope: &Scope,
    list: Variable,
    index: Variable,
    out: Variable,
    unroll_factor: usize,
    kernel_name: &str,
) {
    scope.register_type::<ElemA>(list.ty.storage_type());
    scope.register_size::<SizeA>(list.ty.vector_size());
    let len = expand_buffer_length_native(scope, list);
    let tensor = list.into();
    let index = validate_index::expand::<ElemA, SizeA>(
        scope,
        &tensor,
        index.into(),
        len.into(),
        unroll_factor,
        kernel_name.to_string(),
    );
    let ptr = index_expand(scope, list, index.expand, None, false);
    scope.register(Instruction::new(Operation::Copy(ptr), out));
}
