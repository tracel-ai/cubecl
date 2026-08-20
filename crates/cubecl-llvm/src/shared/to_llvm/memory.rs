use crate::shared::to_llvm::{constant::constant_op, ty::scalar_alignment};

use super::prelude::*;
use cubecl_core::ir::{
    dialect::memory::{DeclareVariableOp, IndexOp, LoadOp, StoreOp},
    types::barrier::BarrierType,
};
use pliron::{basic_block::BasicBlock, builtin::ops::FuncOp, irbuild::inserter::OpInsertionPoint};

#[op_interface_impl]
impl ToLLVMDialect for DeclareVariableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value_ty = self.value_ty(ctx).get_type(ctx);

        if value_ty.deref(ctx).is::<BarrierType>() {
            // Only there to replace by something easy to optimize out
            let useless = insert_i32_const(ctx, rewriter, 0);
            rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![useless]);
            return Ok(());
        }

        let elem_ty = cube_type_to_llvm(ctx, value_ty);

        let entry_block = enclosing_entry_block(ctx, self.get_operation());
        let insertion_point = rewriter.get_insertion_point();
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(entry_block));

        let size = insert_i32_const(ctx, rewriter, 1);

        let alloca = llvm::AllocaOp::new(ctx, elem_ty, size);
        let size_op = size.defining_op().expect("constant defines its result");
        rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(size_op));
        rewriter.insert_op(ctx, &alloca);

        rewriter.set_insertion_point(insertion_point);

        let initializer = self.initializer(ctx).map(|initializer| initializer.clone());
        if let Some(initializer) = initializer {
            let constant = constant_op(ctx, initializer);
            rewriter.insert_operation(ctx, constant);

            let store = llvm::StoreOp::new(ctx, constant.result(ctx), alloca.get_result(ctx));
            store.set_alignment(ctx, self.alignment(ctx).0 as u32);
            rewriter.insert_op(ctx, &store);
        }

        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![alloca.get_result(ctx)],
        );
        Ok(())
    }
}

/// The entry block of the function `op` lives in.
fn enclosing_entry_block(ctx: &Context, op: Ptr<Operation>) -> Ptr<BasicBlock> {
    let mut op = op;
    loop {
        op = op
            .deref(ctx)
            .get_parent_op(ctx)
            .expect("declaration must be inside a function");
        if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
            return func.get_entry_block(ctx);
        }
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
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let result = self.get_result(ctx);
        let res_cube_ty = operands_info
            .lookup_most_recent_type(result)
            .unwrap_or_else(|| result.get_type(ctx));
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
