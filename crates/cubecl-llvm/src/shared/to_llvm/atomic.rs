use crate::shared::polyfill::ordered_atomic::{
    OrderedAtomicFetchAddOp, OrderedAtomicLoadOp, OrderedAtomicStoreOp,
};
use crate::shared::to_llvm::ty::scalar_alignment;

use super::prelude::*;
use cubecl_core::ir::dialect::atomic::*;
use pliron_llvm::attributes::{AtomicOrderingAttr, AtomicRmwKindAttr};

macro_rules! lower_atomic_rmw {
    ($cube_op:ty => $pred:expr) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let ptr = self.ptr(ctx);
                let val = self.value(ctx);
                let kind = $pred;
                let ordering = AtomicOrderingAttr::Monotonic;

                let op = llvm::AtomicRmwOp::new(ctx, ptr, val, kind, ordering, None);
                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_atomic_rmw!(AtomicExchangeOp => AtomicRmwKindAttr::Xchg);
lower_atomic_rmw!(AtomicIAddOp => AtomicRmwKindAttr::Add);
lower_atomic_rmw!(AtomicFAddOp => AtomicRmwKindAttr::FAdd);
lower_atomic_rmw!(AtomicISubOp => AtomicRmwKindAttr::Sub);
lower_atomic_rmw!(AtomicFSubOp => AtomicRmwKindAttr::FSub);
lower_atomic_rmw!(AtomicSMinOp => AtomicRmwKindAttr::Min);
lower_atomic_rmw!(AtomicUMinOp => AtomicRmwKindAttr::UMin);
lower_atomic_rmw!(AtomicFMinOp => AtomicRmwKindAttr::FMin);
lower_atomic_rmw!(AtomicSMaxOp => AtomicRmwKindAttr::Max);
lower_atomic_rmw!(AtomicUMaxOp => AtomicRmwKindAttr::UMax);
lower_atomic_rmw!(AtomicFMaxOp => AtomicRmwKindAttr::FMax);
lower_atomic_rmw!(AtomicAndOp => AtomicRmwKindAttr::And);
lower_atomic_rmw!(AtomicOrOp => AtomicRmwKindAttr::Or);
lower_atomic_rmw!(AtomicXorOp => AtomicRmwKindAttr::Xor);

#[op_interface_impl]
impl ToLLVMDialect for AtomicLoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let ordering = AtomicOrderingAttr::Monotonic;
        let result = self.get_result(ctx);
        let res_cube_ty = operands_info
            .lookup_most_recent_type(result)
            .unwrap_or_else(|| result.get_type(ctx));
        let align = scalar_alignment(ctx, res_cube_ty);
        let res_ty = cube_type_to_llvm(ctx, res_cube_ty);

        let op = llvm::AtomicLoadOp::new(ctx, ptr, res_ty, ordering, None);
        op.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for AtomicStoreOp {
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
        let ordering = AtomicOrderingAttr::Monotonic;

        let store = llvm::AtomicStoreOp::new(ctx, value, ptr, ordering, None);
        store.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &store);
        rewriter.replace_operation(ctx, self.get_operation(), store.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for OrderedAtomicLoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let ordering = self.ordering(ctx).clone();
        let result = self.get_result(ctx);
        let res_cube_ty = operands_info
            .lookup_most_recent_type(result)
            .unwrap_or_else(|| result.get_type(ctx));
        let align = scalar_alignment(ctx, res_cube_ty);
        let res_ty = cube_type_to_llvm(ctx, res_cube_ty);

        let op = llvm::AtomicLoadOp::new(ctx, ptr, res_ty, ordering, None);
        op.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for OrderedAtomicStoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let ordering = self.ordering(ctx).clone();
        let value_cube_ty = operands_info
            .lookup_most_recent_type(value)
            .unwrap_or_else(|| value.get_type(ctx));
        let align = scalar_alignment(ctx, value_cube_ty);

        let store = llvm::AtomicStoreOp::new(ctx, value, ptr, ordering, None);
        store.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &store);
        rewriter.replace_operation(ctx, self.get_operation(), store.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for OrderedAtomicFetchAddOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let ordering = self.ordering(ctx).clone();

        let op = llvm::AtomicRmwOp::new(ctx, ptr, value, AtomicRmwKindAttr::Add, ordering, None);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for AtomicCompareExchangeWeakOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let cmp = self.cmp(ctx);
        let new_val = self.value(ctx);
        let before = AtomicOrderingAttr::Monotonic;
        let after = AtomicOrderingAttr::Monotonic;

        let op = llvm::AtomicCmpxchgOp::new(ctx, ptr, cmp, new_val, before, after, None);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}
