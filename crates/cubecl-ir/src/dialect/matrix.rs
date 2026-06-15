use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::{
    derive::{op_interface_impl, pliron_attr},
    r#type::TypePtr,
};

use crate::{
    attributes::{BoolAttr, IndexAttr},
    dialect::synchronization::SyncScope,
    interfaces::{ReadsMemory, synchronizes},
    pliron::prelude::*,
    types::{
        MatrixShape,
        matrix::{MatrixLayout, MatrixType},
        scalar::UIntType,
    },
};

#[pliron_attr(name = "matrix.layout", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord)]
pub struct MatrixLayoutAttr(pub MatrixLayout);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq)]
pub struct MatrixTypeAttr(pub TypePtr<MatrixType>);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq)]
pub struct MatrixShapeAttr(pub MatrixShape);

/// Fill a matrix with a scalar value.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.fill")]
#[result_ty(none)]
pub struct FillOp {
    #[operand(ptr_write)]
    matrix: Value,
    value: Value,
}

#[cube_op(name = "matrix.load")]
#[result_ty(none)]
pub struct LoadOp {
    #[operand(ptr_write)]
    matrix: Value,
    #[operand(ptr_read)]
    source: Value,
    stride: Value,
    #[attribute(optional)]
    layout: MatrixLayoutAttr,
}
synchronizes!(LoadOp, SyncScope::Plane);

#[cube_op(name = "matrix.store")]
#[result_ty(none)]
pub struct StoreOp {
    #[operand(ptr_read)]
    matrix: Value,
    #[operand(ptr_write)]
    destination: Value,
    stride: Value,
    layout: MatrixLayoutAttr,
}
synchronizes!(StoreOp, SyncScope::Plane);

#[cube_op(name = "matrix.multiply_accumulate")]
#[result_ty(none)]
pub struct MultiplyAccumulateOp {
    mat_a: Value,
    mat_b: Value,
    mat_c: Value,
    #[operand(ptr_write)]
    mat_d: Value,
}
synchronizes!(MultiplyAccumulateOp, SyncScope::Plane);

/// Cast a matrix from one type to another.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.cast")]
#[result_ty(none)]
pub struct CastOp {
    #[operand(ptr_read)]
    input: Value,
    #[operand(ptr_write)]
    output: Value,
}

#[cube_op(name = "matrix.row_index")]
#[result_ty(fixed = UIntType::get(ctx, 32).into())]
pub struct RowIndexOp {
    lane_id: Value,
    i: Value,
    matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.col_index")]
#[result_ty(fixed = UIntType::get(ctx, 32).into())]
pub struct ColIndexOp {
    lane_id: Value,
    i: Value,
    matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.ldmatrix")]
#[result_ty(none)]
pub struct LdMatrixOp {
    ptr: Value,
    out_arr: Value,
    factor: IndexAttr,
    transpose: BoolAttr,
}
synchronizes!(LdMatrixOp, SyncScope::Plane);

#[op_interface_impl]
impl ReadsMemory for LdMatrixOp {
    fn reads_through_values(&self, ctx: &Context) -> Vec<Value> {
        vec![self.ptr(ctx)]
    }
}

#[cube_op(name = "matrix.stmatrix")]
#[result_ty(none)]
pub struct StMatrixOp {
    registers: Value,
    #[operand(ptr_write)]
    destination: Value,
    factor: IndexAttr,
    transpose: BoolAttr,
}
synchronizes!(StMatrixOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual")]
#[result_ty(none)]
pub struct MmaManualOp {
    registers_a: Value,
    registers_b: Value,
    registers_c: Value,
    registers_d: Value,
    shape: MatrixShapeAttr,
}
synchronizes!(MmaManualOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual_scaled")]
#[result_ty(none)]
pub struct MmaManualScaledOp {
    registers_a: Value,
    registers_b: Value,
    registers_c: Value,
    registers_d: Value,
    scales_a: Value,
    scales_b: Value,
    scales_factor: IndexAttr,
    shape: MatrixShapeAttr,
}
synchronizes!(MmaManualScaledOp, SyncScope::Plane);

/// Executes a closure for each element in the matrix.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.elementwise")]
#[result_ty(none)]
pub struct ElementwiseOp {
    matrix_in: Value,
    matrix_out: Value,
    closure: IndexAttr,
}
