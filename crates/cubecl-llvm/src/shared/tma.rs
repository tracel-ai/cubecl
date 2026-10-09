//! Target-specific tensor memory accelerator lowering, and the tensor map type.

use crate::{prelude::*, shared::to_llvm::nvptx_only};
use cubecl_core::ir::{
    dialect::tma::{
        CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupOp, WaitGroupReadOp,
    },
    types::cuda::TensorMapType,
};

/// A tensor map is passed by value, and the kernel holds its address in the generic space.
#[type_interface_impl]
impl CubeToLLVMType for TensorMapType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        LlvmPointerType::get(ctx, 0).into()
    }
}

nvptx_only!(TmaLoadOp, tma::load);
nvptx_only!(TmaLoadIm2colOp, tma::load_im2col);
nvptx_only!(TmaStoreOp, tma::store);
nvptx_only!(CommitGroupOp, tma::commit_group);
nvptx_only!(WaitGroupOp, tma::wait_group);
nvptx_only!(WaitGroupReadOp, tma::wait_group_read);
