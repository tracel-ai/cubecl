//! Target-specific tensor memory accelerator lowering, and the tensor map type.

use crate::{prelude::*, shared::to_llvm::nvptx_only};
use cubecl_core::ir::{
    dialect::tma::{CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupReadOp},
    types::cuda::TensorMapType,
};

/// A tensor map is passed by value, and the kernel holds its generic address.
#[type_interface_impl]
impl CubeToLLVMType for TensorMapType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        LlvmPointerType::get(ctx, GENERIC_ADDRESS_SPACE).into()
    }
}

/// The generic address space, the same on every GPU target.
const GENERIC_ADDRESS_SPACE: u32 = 0;

nvptx_only!(TmaLoadOp, tma::load);
nvptx_only!(TmaLoadIm2colOp, tma::load_im2col);
nvptx_only!(TmaStoreOp, tma::store);
nvptx_only!(CommitGroupOp, tma::commit_group);
nvptx_only!(WaitGroupReadOp, tma::wait_group_read);
