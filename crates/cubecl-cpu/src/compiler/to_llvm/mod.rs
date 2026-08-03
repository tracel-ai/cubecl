pub mod atomic;
pub mod cmp;
pub mod constant;
pub mod general;
pub mod math;
pub mod memory;
pub mod synchronization;
pub mod ty;
pub mod vector;

use prelude::*;

pub mod prelude {
    pub use super::constant::{
        I32_WIDTH, convert_attr, float_attr, insert_bool_const, insert_i32_const, insert_int_const,
        int_attr,
    };
    pub use super::ty::{INDEX_WIDTH, cube_type_to_llvm, llvm_mangled_ty};
    pub use super::vector::insert_splat;
    pub use super::{CubeToLLVMType, ToLLVMDialect};

    pub use cubecl_core::ir::interfaces::{AlignedType, ScalarizableType, TypedExt};
    pub use cubecl_core::ir::prelude::*;
    pub use cubecl_core::ir::types::scalar::{BoolType, IndexType};
    pub use cubecl_core::ir::types::{
        ArrayType as CubeArrayType, PointerType as CubePointerType, VectorType as CubeVectorType,
    };

    pub use pliron::attribute::AttrObj;
    pub use pliron::builtin::attributes::{FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr};
    pub use pliron::builtin::types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness};
    pub use pliron::utils::apint::{APInt, bw};

    pub use pliron_llvm::attributes::{
        FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr, IntegerOverflowFlagsAttr,
    };
    pub use pliron_llvm::op_interfaces::{
        AlignableOpInterface, BinArithOp, CastOpInterface, CastOpWithNNegInterface, FastMathFlags,
        FloatBinArithOpWithFastMathFlags, IntBinArithOpWithOverflowFlag,
    };
    pub use pliron_llvm::ops as llvm;
    pub use pliron_llvm::types::{
        FuncType, PointerType as LlvmPointerType, VectorType as LlvmVectorType, VectorTypeKind,
    };
}

#[type_interface]
pub trait CubeToLLVMType {
    verify_ty_succ!();
    /// Build the LLVM-dialect type equivalent to this cube type.
    fn convert(&self, ctx: &Context) -> TypeHandle;
}

#[op_interface]
pub trait ToLLVMDialect {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

pub type CubeToLLVMPass = DialectConversionPass<CubeToLLVM>;

#[derive(Default)]
pub struct CubeToLLVM;

impl DialectConversion for CubeToLLVM {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToLLVMDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_cast::<dyn CubeToLLVMType>(&*ty.deref(ctx)).is_some()
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let ty = ty.deref(ctx);
        let ty = type_cast::<dyn CubeToLLVMType>(&*ty).unwrap();
        let ty = ty.convert(ctx);
        Ok(ty)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_llvm_op = op_cast::<dyn ToLLVMDialect>(&*op_dyn)
            .expect("Matched Op must implement ToLLVMDialect");
        to_llvm_op.rewrite(ctx, rewriter, operands_info)
    }
}
