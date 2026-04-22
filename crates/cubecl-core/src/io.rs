use alloc::{
    borrow::Cow,
    string::{String, ToString},
};
use derive_more::Display;

use crate as cubecl;
use cubecl::prelude::*;
use cubecl_ir::{Instruction, Operation, Variable};

define_scalar!(ElemA);
define_size!(SizeA);

/// Returns the value at `index` in `list` if `condition` is `true`, otherwise returns `value`.
#[cube]
pub fn read_masked<C: CubePrimitive>(mask: bool, list: &Slice<C>, index: usize, value: C) -> C {
    let index = index * usize::cast_from(mask);
    let input = *list.read_unchecked(index);

    select(mask, input, value)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn checked_index<E: Scalar, N: Size>(
    tensor: &Array<Vector<E, N>>,
    index: usize,
    #[comptime] unroll_factor: usize,
) -> &Vector<E, N> {
    let len = tensor.buffer_len() * unroll_factor;
    let index = index.min(len - 1);

    tensor.read_unchecked(index)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn validate_index<E: Scalar, N: Size>(
    tensor: &Array<Vector<E, N>>,
    index: usize,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) -> &Vector<E, N> {
    let len = tensor.buffer_len() * unroll_factor;
    let in_bounds = index < len;
    if !in_bounds {
        print_oob::<Array<Vector<E, N>>>(kernel_name, OobKind::Read, index, len, tensor);
    }

    let index = index.min(len - 1);

    tensor.read_unchecked(index)
}

#[cube]
fn checked_index_mut<E: Scalar, N: Size>(
    out: &mut Array<Vector<E, N>>,
    index: usize,
    #[comptime] has_buffer_len: bool,
    #[comptime] unroll_factor: usize,
) -> &mut Vector<E, N> {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };
    let len = array_len * unroll_factor;

    let index = index.min(len - 1);

    unsafe { out.index_mut_unchecked(index) }
}

#[cube]
fn validate_index_mut<E: Scalar, N: Size>(
    out: &mut Array<Vector<E, N>>,
    index: usize,
    #[comptime] has_buffer_len: bool,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) -> &mut Vector<E, N> {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };
    let len = array_len * unroll_factor;

    let clamped_index = index.min(len - 1);

    if index >= len {
        print_oob::<Array<Vector<E, N>>>(kernel_name, OobKind::Write, index, len, &*out);
    }
    unsafe { out.index_mut_unchecked(clamped_index) }
}

#[derive(Display)]
enum OobKind {
    #[display("read")]
    Read,
    #[display("write")]
    Write,
}

#[cube]
#[allow(unused)]
fn print_oob<Out: CubeType<ExpandType: Clone + Into<Variable>>>(
    #[comptime] kernel_name: String,
    #[comptime] kind: OobKind,
    index: usize,
    len: usize,
    buffer: &Out,
) {
    intrinsic!(|scope| {
        let name = name_of_var(scope, buffer.clone().into());
        __expand_debug_print!(
            scope,
            alloc::format!(
                "[VALIDATION {kernel_name}]: Encountered OOB {kind} in {name} at %u, length is %u\n"
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
    let array = list.into();
    let ptr = checked_index::expand::<ElemA, SizeA>(scope, &array, index.into(), unroll_factor);
    scope.register(Instruction::new(Operation::Copy(ptr.expand), out));
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
    let array = list.into();
    let ptr = validate_index::expand::<ElemA, SizeA>(
        scope,
        &array,
        index.into(),
        unroll_factor,
        kernel_name.to_string(),
    );
    scope.register(Instruction::new(Operation::Copy(ptr.expand), out));
}

#[allow(missing_docs)]
pub fn expand_checked_index_mut(
    scope: &Scope,
    list: Variable,
    index: Variable,
    out: Variable,
    unroll_factor: usize,
) {
    scope.register_type::<ElemA>(list.ty.storage_type());
    scope.register_size::<SizeA>(list.ty.vector_size());
    let mut array = list.into();
    let ptr = checked_index_mut::expand::<ElemA, SizeA>(
        scope,
        &mut array,
        index.into(),
        list.has_buffer_length(),
        unroll_factor,
    );
    scope.register(Instruction::new(Operation::Copy(ptr.expand), out));
}

#[allow(missing_docs)]
pub fn expand_validate_index_mut(
    scope: &Scope,
    list: Variable,
    index: Variable,
    out: Variable,
    unroll_factor: usize,
    kernel_name: &str,
) {
    scope.register_type::<ElemA>(list.ty.storage_type());
    scope.register_size::<SizeA>(list.ty.vector_size());
    let mut array = list.into();
    let ptr = validate_index_mut::expand::<ElemA, SizeA>(
        scope,
        &mut array,
        index.into(),
        list.has_buffer_length(),
        unroll_factor,
        kernel_name.to_string(),
    );
    scope.register(Instruction::new(Operation::Copy(ptr.expand), out));
}
