use cubecl_core::ir::dialect::memory::{DeclareVariableOp, IndexOp, LoadOp, StoreOp};
use cubecl_core::ir::types::{ArrayType as CubeArrayType, PointerType as CubePointerType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::ops::ConstantOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::utils::apint::{APInt, bw};
use pliron_llvm::ops as llvm;

use crate::compiler::dialect::ty::cube_type_to_llvm;

use super::prelude::*;

#[op_interface_impl]
impl ToLLVMDialect for DeclareVariableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value_ty = self.value_ty(ctx).get_type(ctx);

        let (elem_ty, count) = {
            let value_ty = value_ty.deref(ctx);
            match value_ty.downcast_ref::<CubeArrayType>() {
                Some(array) => (array.inner, array.length),
                None => (self.value_ty(ctx).get_type(ctx), 1),
            }
        };
        let elem_ty = cube_type_to_llvm(ctx, elem_ty);

        let size_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let size_attr = IntegerAttr::new(size_ty, APInt::from_u32(count as u32, bw(32)));
        let size = ConstantOp::new(ctx, size_attr.into());
        rewriter.insert_op(ctx, &size);

        let alloca = llvm::AllocaOp::new(ctx, elem_ty, size.get_result(ctx));
        rewriter.insert_op(ctx, &alloca);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![alloca.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for IndexOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = self.base(ctx);
        let index = self.index(ctx);

        let elem_ty = {
            let result_ty = self.get_result(ctx).get_type(ctx);
            result_ty
                .deref(ctx)
                .downcast_ref::<CubePointerType>()
                .expect("memory.index result must be a pointer")
                .inner
        };
        let elem_ty = cube_type_to_llvm(ctx, elem_ty);

        let gep =
            llvm::GetElementPtrOp::new(ctx, base, vec![llvm::GepIndex::Value(index)], elem_ty);
        rewriter.insert_op(ctx, &gep);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![gep.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for LoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let res_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));

        let load = llvm::LoadOp::new(ctx, ptr, res_ty);
        rewriter.insert_op(ctx, &load);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![load.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for StoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        let ptr = self.ptr(ctx);

        let store = llvm::StoreOp::new(ctx, value, ptr);
        rewriter.insert_op(ctx, &store);
        rewriter.replace_operation(ctx, self.get_operation(), store.get_operation());
        Ok(())
    }
}
