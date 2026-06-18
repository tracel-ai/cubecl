use cubecl_macros_internal::cube_op;
use derive_more::{Deref, From};
use derive_new::new;
use pliron::{
    derive::{op_interface_impl, pliron_attr},
    r#type::TypedHandle,
};

use crate::{
    attributes::{BoolAttr, IndexAttr},
    dialect::synchronization::SyncScope,
    interfaces::{ReadsMemory, synchronizes},
    prelude::*,
    types::{
        MatrixShape,
        matrix::{MatrixLayout, MatrixType},
        scalar::UIntType,
    },
};

#[pliron_attr(name = "matrix.layout", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord, Deref)]
pub struct MatrixLayoutAttr(pub MatrixLayout);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq, Deref)]
pub struct MatrixTypeAttr(pub TypedHandle<MatrixType>);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq, Deref)]
pub struct MatrixShapeAttr(pub MatrixShape);

/// Fill a matrix with a scalar value.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.fill")]
#[result_ty(none)]
pub struct FillOp {
    #[operand(ptr_write)]
    pub matrix: Value,
    pub value: Value,
}

#[cube_op(name = "matrix.load")]
#[result_ty(none)]
pub struct LoadOp {
    #[operand(ptr_write)]
    pub matrix: Value,
    #[operand(ptr_read)]
    pub source: Value,
    pub stride: Value,
    #[attribute(optional)]
    pub layout: MatrixLayoutAttr,
}
synchronizes!(LoadOp, SyncScope::Plane);

#[cube_op(name = "matrix.store")]
#[result_ty(none)]
pub struct StoreOp {
    #[operand(ptr_read)]
    pub matrix: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub stride: Value,
    pub layout: MatrixLayoutAttr,
}
synchronizes!(StoreOp, SyncScope::Plane);

#[cube_op(name = "matrix.multiply_accumulate")]
#[result_ty(none)]
pub struct MultiplyAccumulateOp {
    pub mat_a: Value,
    pub mat_b: Value,
    pub mat_c: Value,
    #[operand(ptr_write)]
    pub mat_d: Value,
}
synchronizes!(MultiplyAccumulateOp, SyncScope::Plane);

/// Cast a matrix from one type to another.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.cast")]
#[result_ty(none)]
pub struct CastOp {
    #[operand(ptr_read)]
    pub input: Value,
    #[operand(ptr_write)]
    pub output: Value,
}

#[cube_op(name = "matrix.row_index")]
#[result_ty(fixed = UIntType::get(ctx, 32).into())]
pub struct RowIndexOp {
    pub lane_id: Value,
    pub i: Value,
    pub matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.col_index")]
#[result_ty(fixed = UIntType::get(ctx, 32).into())]
pub struct ColIndexOp {
    pub lane_id: Value,
    pub i: Value,
    pub matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.ldmatrix")]
#[result_ty(none)]
pub struct LdMatrixOp {
    pub ptr: Value,
    pub out_arr: Value,
    pub factor: IndexAttr,
    pub transpose: BoolAttr,
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
    pub registers: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub factor: IndexAttr,
    pub transpose: BoolAttr,
}
synchronizes!(StMatrixOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual")]
#[result_ty(none)]
pub struct MmaManualOp {
    pub registers_a: Value,
    pub registers_b: Value,
    pub registers_c: Value,
    pub registers_d: Value,
    pub shape: MatrixShapeAttr,
}
synchronizes!(MmaManualOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual_scaled")]
#[result_ty(none)]
pub struct MmaManualScaledOp {
    pub registers_a: Value,
    pub registers_b: Value,
    pub registers_c: Value,
    pub registers_d: Value,
    pub scales_a: Value,
    pub scales_b: Value,
    pub scales_factor: IndexAttr,
    pub shape: MatrixShapeAttr,
}
synchronizes!(MmaManualScaledOp, SyncScope::Plane);

/// Executes a closure for each element in the matrix.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.elementwise")]
#[result_ty(none)]
pub struct ElementwiseOp {
    pub matrix_in: Value,
    pub matrix_out: Value,
    pub closure: IndexAttr,
}
