//! Buffers of `#[repr(C)]` structs ("records").
//!
//! A buffer of records `T` is a runtime array of `array<u32, size_of::<T>() / 4>`. Indexing it
//! gives a pointer to one record; each field is a pointer at a constant word offset from it,
//! cast to the field's type. The offsets come from `core::mem::offset_of!`, so the host and the
//! kernel share the layout `rustc` chose for the `#[repr(C)]` struct.

use cubecl_ir::{
    dialect::{general::ReinterpretCastOp, memory::IndexOp},
    pliron::r#type::Typed,
    types::{ArrayType, PointerType},
};

use crate::prelude::*;

/// A value stored inside a buffer record, at a word (4 byte) offset.
///
/// Implemented for `f32`, `u32`, `i32`, `f64`, `u64` and `i64`, and by `#[derive(CubeType)]`
/// for `#[repr(C)]` structs, which are then also [`SliceElement`]s.
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be a field of a buffer record",
    note = "record fields are f32, u32, i32, f64, u64, i64, or #[repr(C)] structs deriving CubeType"
)]
pub trait RecordField: CubeType {
    /// The places of this value's parts, `word` words into the record `rec` points to.
    #[doc(hidden)]
    fn __expand_record_field(scope: &Scope, rec: ElemPtr, word: usize) -> Self::ExpandType;
}

/// The place of a scalar field: word `word` of the record, cast to a pointer to `F`.
fn scalar_field<F: CubePrimitive>(scope: &Scope, rec: ElemPtr, word: usize) -> NativeExpand<F> {
    let index: NativeExpand<usize> = word.into_expand(scope);
    let index = index.read_value(scope);
    let op = IndexOp::maybe_checked(scope.ctx_mut(), rec.ptr, index, false);
    let word_ptr = scope.register_with_result(&op);

    let PointerType {
        inner,
        address_space,
    } = {
        let ctx = scope.ctx();
        let ty = word_ptr.get_type(ctx).deref(ctx);
        *ty.downcast_ref()
            .expect("a record word should be a pointer")
    };
    let field_ty = F::__expand_as_type(scope);
    if inner == field_ty {
        return word_ptr.into();
    }
    let ptr_ty = PointerType::get(scope.ctx(), field_ty, address_space).into();
    let cast = ReinterpretCastOp::new(scope.ctx_mut(), ptr_ty, word_ptr);
    scope.register_with_result(&cast).into()
}

macro_rules! impl_record_field {
    ($($ty:ty),*) => {$(
        impl RecordField for $ty {
            fn __expand_record_field(scope: &Scope, rec: ElemPtr, word: usize) -> NativeExpand<$ty> {
                scalar_field::<$ty>(scope, rec, word)
            }
        }
    )*};
}

impl_record_field!(f32, u32, i32, f64, u64, i64);

impl BufferElem {
    /// The buffer element of a record of `size` bytes: `array<u32, size / 4>`.
    #[doc(hidden)]
    pub fn record(scope: &Scope, size: usize) -> Self {
        let word = u32::__expand_as_type(scope);
        BufferElem {
            ty: ArrayType::get(scope.ctx(), word, size / 4).into(),
            size,
        }
    }
}
