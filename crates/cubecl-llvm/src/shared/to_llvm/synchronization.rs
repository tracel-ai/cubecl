use super::prelude::*;
use crate::shared::polyfill::synchronization::SpinLoopHintOp;
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
