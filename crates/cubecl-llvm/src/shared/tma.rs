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

#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; only NVPTX lowers TMA and the barrier copies, \
     and only it advertises barriers"
)]
pub struct TmaUnsupported(LlvmTarget, &'static str);

/// A tensor map is passed by value, and the kernel holds its address.
#[type_interface_impl]
impl CubeToLLVMType for TensorMapType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        LlvmPointerType::get(ctx, 0).into()
    }
}

/// The barrier state an arrival returns, for the wait on its phase.
#[type_interface_impl]
impl CubeToLLVMType for BarrierTokenType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, 64, Signedness::Signless).into()
    }
}

/// TMA is Hopper's, so only NVPTX lowers it.
macro_rules! dispatch_tma_op {
    ($cube_op:ty, $method:ident) => {
        lower_by_target!(
            $cube_op,
            ["nvptx" Nvptx => crate::nvptx::tma::$method],
            |op: &$cube_op, ctx: &mut Context, _: &mut DialectConversionRewriter, target| -> Result<()> {
                input_err!(op.loc(ctx), TmaUnsupported(target, stringify!($cube_op)))
            }
        );
    };
}

/// `memcpy_async` lowers with the barriers.
macro_rules! dispatch_barrier_copy {
    ($cube_op:ty, $method:ident) => {
        lower_by_target!(
            $cube_op,
            ["nvptx" Nvptx => crate::nvptx::barrier::$method],
            |op: &$cube_op, ctx: &mut Context, _: &mut DialectConversionRewriter, target| -> Result<()> {
                input_err!(op.loc(ctx), TmaUnsupported(target, stringify!($cube_op)))
            }
        );
    };
}

dispatch_tma_op!(TmaLoadOp, load);
dispatch_tma_op!(TmaLoadIm2colOp, load_im2col);
dispatch_tma_op!(TmaStoreOp, store);
dispatch_tma_op!(MemCopyAsyncTxOp, memcpy_async_tx);
dispatch_barrier_copy!(MemCopyAsyncOp, memcpy_async);
dispatch_tma_op!(CommitGroupOp, commit_group);
dispatch_tma_op!(WaitGroupOp, wait_group);
dispatch_tma_op!(WaitGroupReadOp, wait_group_read);
