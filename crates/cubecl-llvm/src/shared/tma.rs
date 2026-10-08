//! Target-specific tensor memory accelerator lowering, and the types it passes around.

use crate::{prelude::*, shared::to_llvm::lower_by_target};
use cubecl_core::ir::{
    dialect::{
        barrier::{MemCopyAsyncOp, MemCopyAsyncTxOp},
        tma::{
            CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupOp, WaitGroupReadOp,
        },
    },
    types::{barrier::BarrierTokenType, cuda::TensorMapType},
};

/// The operations Hopper introduced, which only NVPTX lowers and advertises: TMA and the
/// barriers its copies complete on.
#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; only NVPTX lowers TMA and the copies that \
     complete on its barriers, and only it advertises them"
)]
pub struct NvptxOnly(LlvmTarget, &'static str);

/// A tensor map is passed by value, and the kernel holds its generic address.
#[type_interface_impl]
impl CubeToLLVMType for TensorMapType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        LlvmPointerType::get(ctx, GENERIC_ADDRESS_SPACE).into()
    }
}

/// The generic address space, the same on every GPU target.
const GENERIC_ADDRESS_SPACE: u32 = 0;

/// The barrier state an arrival returns, for the wait on its phase.
#[type_interface_impl]
impl CubeToLLVMType for BarrierTokenType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, 64, Signedness::Signless).into()
    }
}

/// Lowers `$cube_op` with `$module::$method` on NVPTX, and refuses it on the other targets.
macro_rules! nvptx_only {
    ($cube_op:ty, $module:ident::$method:ident) => {
        lower_by_target!(
            $cube_op,
            ["nvptx" Nvptx => crate::nvptx::$module::$method],
            |op: &$cube_op, ctx: &mut Context, _: &mut DialectConversionRewriter, target| -> Result<()> {
                input_err!(op.loc(ctx), NvptxOnly(target, stringify!($cube_op)))
            }
        );
    };
}

nvptx_only!(TmaLoadOp, tma::load);
nvptx_only!(TmaLoadIm2colOp, tma::load_im2col);
nvptx_only!(TmaStoreOp, tma::store);
nvptx_only!(MemCopyAsyncTxOp, barrier::memcpy_async_tx);
nvptx_only!(MemCopyAsyncOp, barrier::memcpy_async);
nvptx_only!(CommitGroupOp, tma::commit_group);
nvptx_only!(WaitGroupOp, tma::wait_group);
nvptx_only!(WaitGroupReadOp, tma::wait_group_read);
