//! LLVM-dialect lowering of the CPU-private ops.
//!
//! The ops the CPU pipeline introduces for itself — the spin loop hint of
//! [`synchronization`](super::synchronization) and the explicitly ordered atomics of
//! [`ordered_atomic`](super::ordered_atomic) — become their `llvm` dialect counterparts here,
//! the same way the target independent ops do in [`to_llvm`](crate::shared::to_llvm).

use crate::cpu::ordered_atomic::{
    OrderedAtomicFetchAddOp, OrderedAtomicLoadOp, OrderedAtomicStoreOp,
};
use crate::cpu::synchronization::SpinLoopHintOp;
use crate::shared::to_llvm::prelude::*;
use crate::shared::to_llvm::ty::scalar_alignment;

use pliron_llvm::attributes::AtomicRmwKindAttr;
use pliron_llvm::types::VoidType;

/// The instruction hinting the core that it is in a spin loop, i.e. what `std::hint::spin_loop`
/// emits. The kernel is JIT'd for the host, so the target is simply the one we are built for.
const fn spin_loop_instruction() -> Option<&'static str> {
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        Some("pause")
    } else if cfg!(any(target_arch = "aarch64", target_arch = "arm")) {
        Some("yield")
    } else {
        None
    }
}

#[op_interface_impl]
impl ToLLVMDialect for SpinLoopHintOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        // An architecture without such a hint spins just as correctly, only busier.
        if let Some(instruction) = spin_loop_instruction() {
            let void_ty = VoidType::get(ctx).into();
            let hint = llvm::InlineAsmOp::new(ctx, void_ty, vec![], instruction, "", false);
            rewriter.insert_op(ctx, &hint);
        }
        rewriter.erase_operation(ctx, self.get_operation());
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
