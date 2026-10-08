//! Target-specific tensor memory accelerator lowering, and the types it passes around.

use crate::prelude::*;
use cubecl_core::ir::{
    dialect::{
        barrier::MemCopyAsyncTxOp,
        tma::{
            CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupOp, WaitGroupReadOp,
        },
    },
    types::{barrier::BarrierTokenType, cuda::TensorMapType},
};

#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; TMA is Hopper's, and only NVPTX advertises it"
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

macro_rules! dispatch_tma_op {
    ($cube_op:ty, $method:ident) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                _rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                match ctx.target() {
                    #[cfg(feature = "nvptx")]
                    LlvmTarget::Nvptx => {
                        crate::nvptx::tma::$method(self, ctx, _rewriter, _operands_info)
                    }
                    target => {
                        input_err!(self.loc(ctx), TmaUnsupported(target, stringify!($cube_op)))
                    }
                }
            }
        }
    };
}

dispatch_tma_op!(TmaLoadOp, load);
dispatch_tma_op!(TmaLoadIm2colOp, load_im2col);
dispatch_tma_op!(TmaStoreOp, store);
dispatch_tma_op!(MemCopyAsyncTxOp, memcpy_async_tx);
dispatch_tma_op!(CommitGroupOp, commit_group);
dispatch_tma_op!(WaitGroupOp, wait_group);
dispatch_tma_op!(WaitGroupReadOp, wait_group_read);
