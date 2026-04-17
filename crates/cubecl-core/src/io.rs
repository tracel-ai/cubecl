use alloc::{
    borrow::Cow,
    string::{String, ToString},
};
use derive_more::Display;

use crate as cubecl;
use cubecl::prelude::*;
use cubecl_ir::{ManagedVariable, Variable};

define_scalar!(ElemA);
define_size!(SizeA);

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
    let index = index.min(len - 1);

    select(in_bounds, tensor.read_unchecked(index), C::default())
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_atomic_checked<C: Scalar>(
    tensor: Tensor<Atomic<C>>,
    index: usize,
    #[comptime] unroll_factor: usize,
) -> Atomic<C> {
    let index = index.min(tensor.buffer_len() * unroll_factor - 1);

    tensor.read_unchecked(index)
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_validate<C: CubePrimitive + Default + IntoRuntime>(
    tensor: Tensor<C>,
    index: usize,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) -> C {
    let len = tensor.buffer_len() * unroll_factor;
    let in_bounds = index < len;
    if !in_bounds {
        print_oob::<Tensor<C>>(kernel_name, OobKind::Read, index, len, &tensor);
    }

    let index = index.min(len - 1);

    select(in_bounds, tensor.read_unchecked(index), C::default())
}

/// Returns the value at `index` in tensor within bounds.
#[cube]
pub fn read_tensor_atomic_validate<C: Scalar>(
    tensor: Tensor<Atomic<C>>,
    index: usize,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) -> Atomic<C> {
    let len = tensor.buffer_len() * unroll_factor;
    if index >= len {
        print_oob::<Tensor<Atomic<C>>>(kernel_name, OobKind::Read, index, len, &tensor);
    }
    let index = index.min(tensor.buffer_len() * unroll_factor - 1);

    tensor.read_unchecked(index)
}

#[cube]
fn checked_index_assign<E: Scalar, N: Size>(
    index: usize,
    value: Vector<E, N>,
    out: &mut Array<Vector<E, N>>,
    #[comptime] has_buffer_len: bool,
    #[comptime] unroll_factor: usize,
) {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };

    if index < array_len * unroll_factor {
        unsafe { out.index_assign_unchecked(index, value) };
    }
}

#[cube]
fn validate_index_assign<E: Scalar, N: Size>(
    index: usize,
    value: Vector<E, N>,
    out: &mut Array<Vector<E, N>>,
    #[comptime] has_buffer_len: bool,
    #[comptime] unroll_factor: usize,
    #[comptime] kernel_name: String,
) {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };
    let len = array_len * unroll_factor;

    if index < len {
        unsafe { out.index_assign_unchecked(index, value) };
    } else {
        print_oob::<Array<Vector<E, N>>>(kernel_name, OobKind::Write, index, len, out);
    }
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
fn print_oob<Out: CubeType<ExpandType: Into<Variable>>>(
    #[comptime] kernel_name: String,
    #[comptime] kind: OobKind,
    index: usize,
    len: usize,
    buffer: &Out,
) {
    intrinsic!(|scope| {
        let name = name_of_var(scope, buffer.into());
        debug_print_expand!(
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
pub fn expand_checked_index_assign(
    scope: &mut Scope,
    lhs: Variable,
    rhs: Variable,
    out: Variable,
    unroll_factor: usize,
) {
    scope.register_type::<ElemA>(rhs.ty.storage_type());
    scope.register_size::<SizeA>(rhs.ty.vector_size());
    checked_index_assign::expand::<ElemA, SizeA>(
        scope,
        ManagedVariable::Plain(lhs).into(),
        ManagedVariable::Plain(rhs).into(),
        ManagedVariable::Plain(out).into(),
        out.has_buffer_length(),
        unroll_factor,
    );
}

#[allow(missing_docs)]
pub fn expand_validate_index_assign(
    scope: &mut Scope,
    lhs: Variable,
    rhs: Variable,
    out: Variable,
    unroll_factor: usize,
    kernel_name: &str,
) {
    scope.register_type::<ElemA>(rhs.ty.storage_type());
    scope.register_size::<SizeA>(rhs.ty.vector_size());
    validate_index_assign::expand::<ElemA, SizeA>(
        scope,
        ManagedVariable::Plain(lhs).into(),
        ManagedVariable::Plain(rhs).into(),
        ManagedVariable::Plain(out).into(),
        out.has_buffer_length(),
        unroll_factor,
        kernel_name.to_string(),
    );
}
