use super::ToLLVMDialect;
use cubecl_core::ir::dialect::memory::{DeclareVariableOp, IndexOp, LoadOp, StoreOp};
use cubecl_core::ir::interfaces::{AlignedType, ScalarizableType};
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::types::{ArrayType as CubeArrayType, PointerType as CubePointerType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::ops::ConstantOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::utils::apint::{APInt, bw};
use pliron_llvm::op_interfaces::AlignableOpInterface;
use pliron_llvm::ops as llvm;

use crate::compiler::to_llvm::ty::cube_type_to_llvm;

fn scalar_alignment(ctx: &Context, ty: TypeHandle) -> u32 {
    let scalar = {
        let ty = ty.deref(ctx);
        type_cast::<dyn ScalarizableType>(&*ty).map(|s| s.scalar_type(ctx))
    }
    .unwrap_or(ty);

    let scalar = scalar.deref(ctx);
    type_cast::<dyn AlignedType>(&*scalar)
        .expect("load/store value type must implement AlignedType")
        .align(ctx) as u32
}

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
        let res_cube_ty = self.get_result(ctx).get_type(ctx);
        let align = scalar_alignment(ctx, res_cube_ty);
        let res_ty = cube_type_to_llvm(ctx, res_cube_ty);

        let load = llvm::LoadOp::new(ctx, ptr, res_ty);
        load.set_alignment(ctx, align);
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
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        let ptr = self.ptr(ctx);
        let value_cube_ty = operands_info
            .lookup_most_recent_type(value)
            .unwrap_or_else(|| value.get_type(ctx));
        let align = scalar_alignment(ctx, value_cube_ty);

        let store = llvm::StoreOp::new(ctx, value, ptr);
        store.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &store);
        rewriter.replace_operation(ctx, self.get_operation(), store.get_operation());
        Ok(())
    }
}
