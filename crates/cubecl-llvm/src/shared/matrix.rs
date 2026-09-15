//! Target-specific matrix lowering.

use cubecl_core::ir::Scope;
use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp,
    StMatrixOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::prelude::polyfills;
use cubecl_core::prelude::*;
use pliron::input_err;
use thiserror::Error;

use crate::shared::polyfill::LowerOp;
use crate::shared::to_llvm::prelude::*;
use crate::target::{CtxTarget, LlvmTarget};
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
use cubecl_core::ir::types::ArrayType as CubeArrayType;

#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; a runtime that reaches it here has advertised \
     a matrix feature it cannot honour"
)]
pub struct MatrixOpUnsupported(LlvmTarget, &'static str);

#[type_interface_impl]
impl CubeToLLVMType for MatrixType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        match ctx.target() {
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => crate::amdgpu::matrix::fragment_ty(ctx, self),
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => crate::nvptx::matrix::fragment_ty(ctx, self),
            LlvmTarget::Cpu => {
                unimplemented!("the CPU target has no matrix fragments")
            }
        }
    }
}

macro_rules! dispatch_matrix_op {
    ($cube_op:ty, $method:ident) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                _rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                match ctx.target() {
                    #[cfg(feature = "amdgpu")]
                    LlvmTarget::AmdGpu => {
                        crate::amdgpu::matrix::$method(self, ctx, _rewriter, _operands_info)
                    }
                    #[cfg(feature = "nvptx")]
                    LlvmTarget::Nvptx => {
                        crate::nvptx::matrix::$method(self, ctx, _rewriter, _operands_info)
                    }
                    target => input_err!(
                        self.loc(ctx),
                        MatrixOpUnsupported(target, stringify!($cube_op))
                    ),
                }
            }
        }
    };
}

dispatch_matrix_op!(FillOp, fill);
dispatch_matrix_op!(LoadOp, load);
dispatch_matrix_op!(StoreOp, store);
dispatch_matrix_op!(MultiplyAccumulateOp, multiply_accumulate);
dispatch_matrix_op!(CastOp, cast);
dispatch_matrix_op!(RowIndexOp, row_index);
dispatch_matrix_op!(ColIndexOp, col_index);
dispatch_matrix_op!(MmaManualOp, mma_manual);
dispatch_matrix_op!(LdMatrixOp, ld_matrix);
dispatch_matrix_op!(StMatrixOp, st_matrix);

/// NVPTX matrix coordinates use the shared MMA polyfills.
macro_rules! lower_axis_index_polyfill {
    ($cube_op:ty, $formula:path) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn should_lower(&self, ctx: &Context) -> bool {
                match ctx.target() {
                    #[cfg(feature = "nvptx")]
                    LlvmTarget::Nvptx => true,
                    _ => false,
                }
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
                let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
                let lane_id = self.lane_id(scope.ctx());
                let i = self.i(scope.ctx());

                let index = $formula(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
                vec![index.value(scope)]
            }
        }
    };
}

lower_axis_index_polyfill!(RowIndexOp, polyfills::mma::row_index::expand);
lower_axis_index_polyfill!(ColIndexOp, polyfills::mma::col_index::expand);

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) fn registers_as_vector(
    ctx: &Context,
    info: &OperandsInfo,
    value: Value,
) -> (TypeHandle, TypeHandle) {
    let array = info
        .lookup_operand_history(value)
        .into_iter()
        .rev()
        .chain(core::iter::once(value.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            if let Some(array) = ty.downcast_ref::<CubeArrayType>() {
                return Some(*array);
            }
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let inner = ptr.inner.deref(ctx);
            inner.downcast_ref::<CubeArrayType>().copied()
        })
        .expect("a manual matrix operand is an array of registers");

    let (scalar, per_register) = match array.inner.deref(ctx).downcast_ref::<CubeVectorType>() {
        Some(vector) => (vector.inner, vector.vectorization),
        None => (array.inner, 1),
    };
    let elem = cube_type_to_llvm(ctx, scalar);
    let lanes = (array.length * per_register) as u32;
    let vector = LlvmVectorType::get(ctx, elem, lanes, VectorTypeKind::Fixed).into();
    (vector, scalar)
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) fn registers_value(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    vector_ty: TypeHandle,
) -> Value {
    let ty = value.get_type(ctx);
    if ty.deref(ctx).is::<LlvmPointerType>() {
        let op = llvm::LoadOp::new(ctx, value, vector_ty);
        return insert(ctx, rw, &op);
    }

    let (count, packed) = {
        let ty = ty.deref(ctx);
        let array = ty
            .downcast_ref::<LlvmArrayType>()
            .expect("registers are an array");
        (array.size(), array.elem_type())
    };
    let per_register = match packed.deref(ctx).downcast_ref::<LlvmVectorType>() {
        Some(vector) => vector.num_elements() as u64,
        None => 1,
    };

    let poison = llvm::PoisonOp::new(ctx, vector_ty);
    let mut acc = insert(ctx, rw, &poison);

    for register in 0..count {
        let op = llvm::ExtractValueOp::new(ctx, value, vec![register as u32])
            .expect("a constant index into the register array");
        let element = insert(ctx, rw, &op);

        for lane in 0..per_register {
            let value = if per_register == 1 {
                element
            } else {
                let from = insert_i32_const(ctx, rw, lane as i32);
                let op = llvm::ExtractElementOp::new(ctx, element, from);
                insert(ctx, rw, &op)
            };
            let to = insert_i32_const(ctx, rw, (register * per_register + lane) as i32);
            let op = llvm::InsertElementOp::new(ctx, acc, value, to);
            acc = insert(ctx, rw, &op);
        }
    }
    acc
}

#[cfg(feature = "nvptx")]
pub(crate) fn registers_array_ty(ctx: &Context, info: &OperandsInfo, value: Value) -> TypeHandle {
    let array = info
        .lookup_operand_history(value)
        .into_iter()
        .rev()
        .chain(core::iter::once(value.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            if let Some(array) = ty.downcast_ref::<CubeArrayType>() {
                return Some(*array);
            }
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let inner = ptr.inner.deref(ctx);
            inner.downcast_ref::<CubeArrayType>().copied()
        })
        .expect("a manual matrix operand is an array of registers");
    let elem = cube_type_to_llvm(ctx, array.inner);
    LlvmArrayType::get(ctx, elem, array.length as u64).into()
}

/// Manual matrix outputs must preserve the frontend array type.
#[cfg(feature = "nvptx")]
pub(crate) fn vector_into_array(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    array_ty: TypeHandle,
) -> Value {
    let (count, elem_ty) = {
        let ty = array_ty.deref(ctx);
        let array = ty
            .downcast_ref::<LlvmArrayType>()
            .expect("registers are an array");
        (array.size() as usize, array.elem_type())
    };
    let per_element = match elem_ty.deref(ctx).downcast_ref::<LlvmVectorType>() {
        Some(vector) => vector.num_elements() as usize,
        None => 1,
    };

    let poison = llvm::PoisonOp::new(ctx, array_ty);
    let mut acc = insert(ctx, rw, &poison);

    for index in 0..count {
        let element = if per_element == 1 {
            let at = insert_i32_const(ctx, rw, index as i32);
            let op = llvm::ExtractElementOp::new(ctx, vector, at);
            insert(ctx, rw, &op)
        } else {
            let poison = llvm::PoisonOp::new(ctx, elem_ty);
            let mut packed = insert(ctx, rw, &poison);
            for lane in 0..per_element {
                let from = insert_i32_const(ctx, rw, (index * per_element + lane) as i32);
                let op = llvm::ExtractElementOp::new(ctx, vector, from);
                let value = insert(ctx, rw, &op);
                let to = insert_i32_const(ctx, rw, lane as i32);
                let op = llvm::InsertElementOp::new(ctx, packed, value, to);
                packed = insert(ctx, rw, &op);
            }
            packed
        };
        let op = llvm::InsertValueOp::new(ctx, acc, element, vec![index as u32]);
        acc = insert(ctx, rw, &op);
    }
    acc
}
