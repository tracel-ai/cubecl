//! How an operation picks its lowering by target.

use crate::prelude::*;

/// Implements [`ToLLVMDialect`] for `$cube_op` by its lowering on each target listed, gated on
/// that target's feature, and by `$fallback(op, ctx, rewriter, target)` on the others.
macro_rules! lower_by_target {
    ($cube_op:ty, [$($feature:literal $target:ident => $lower:path),* $(,)?], $fallback:expr) => {
        #[op_interface_impl]
        impl $crate::shared::to_llvm::ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                operands_info: &OperandsInfo,
            ) -> Result<()> {
                let _ = operands_info;
                match ctx.target() {
                    $(
                        #[cfg(feature = $feature)]
                        LlvmTarget::$target => $lower(self, ctx, rewriter, operands_info),
                    )*
                    #[allow(unreachable_patterns)]
                    target => ($fallback)(self, ctx, rewriter, target),
                }
            }
        }
    };
}
pub(crate) use lower_by_target;

/// A [`lower_by_target!`] fallback that drops the operation: a target where it means nothing.
pub(crate) fn erase<O: Op>(
    op: &O,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    _target: LlvmTarget,
) -> Result<()> {
    rewriter.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// The operations Hopper introduced, which only NVPTX lowers and advertises: TMA and the
/// barriers its copies complete on.
#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; only NVPTX lowers TMA and the copies that \
     complete on its barriers, and only it advertises them"
)]
pub struct NvptxOnly(pub(crate) LlvmTarget, pub(crate) &'static str);

/// Lowers `$cube_op` with `$module::$method` on NVPTX, and refuses it on the other targets.
macro_rules! nvptx_only {
    ($cube_op:ty, $module:ident::$method:ident) => {
        $crate::shared::to_llvm::lower_by_target!(
            $cube_op,
            ["nvptx" Nvptx => crate::nvptx::$module::$method],
            |op: &$cube_op, ctx: &mut Context, _: &mut DialectConversionRewriter, target| -> Result<()> {
                input_err!(
                    op.loc(ctx),
                    $crate::shared::to_llvm::NvptxOnly(target, stringify!($cube_op))
                )
            }
        );
    };
}

pub(crate) use nvptx_only;
