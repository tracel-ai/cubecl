//! Buffers of `#[repr(C)]` structs ("records").
//!
//! A buffer of records `T` is a runtime array of `array<u32, size_of::<T>() / 4>`. Indexing it
//! gives a pointer to one record; each field is a pointer at a constant word offset from it,
//! cast to the field's type. The offsets come from `core::mem::offset_of!`, so the host and the
//! kernel share the layout `rustc` chose for the `#[repr(C)]` struct.

use cubecl_ir::{
    ExpandValue,
    dialect::memory::StructFieldOp,
    interfaces::TypeExt,
    types::{self, StructType},
};
use pliron::r#type::Typed;

use crate::prelude::*;

/// A value that has a consistent representation between host and GPU.
/// Implemented for most primitives, but not `usize`, `isize` and `bool` which have a target-specific
/// representation. Also implemented for types that are made up of shareable types and have a
/// GPU-compatible layout (`repr(C)`).
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be shared between CPU and GPU",
    note = "shareable types are are floats, ints, arrays, or #[repr(C)] structs containing shareable fields"
)]
pub trait HostShareable: CubeType + ConstantTyped {
    /// The places of this value's parts
    #[doc(hidden)]
    fn __expand_runtime_place(scope: &Scope, ptr: ExpandValue) -> Self::ExpandType;
}

#[doc(hidden)]
pub fn __field_ptr<T: ConstantTyped>(
    scope: &Scope,
    struct_ptr: ExpandValue,
    field_idx: usize,
) -> ExpandValue {
    let ctx = scope.ctx_mut();
    let struct_ptr = struct_ptr.value(scope);
    let field_ty = T::__expand_as_type(scope);
    let field_ptr_ty = struct_ptr.get_type(ctx).map_ptr(ctx, field_ty);
    let field = StructFieldOp::new(scope.ctx_mut(), field_ptr_ty, struct_ptr, field_idx);
    scope.register_with_result(&field).into()
}

#[doc(hidden)]
pub fn __struct_ty<T: Sized>(scope: &Scope, fields: &[(TypeHandle, usize)]) -> TypeHandle {
    let fields = fields
        .iter()
        .map(|(ty, offset)| types::StructField::new(*ty, *offset as u64))
        .collect();
    let ty = StructType::get(
        scope.ctx(),
        size_of::<T>() as u64,
        align_of::<T>() as u64,
        fields,
    );
    ty.to_handle()
}
