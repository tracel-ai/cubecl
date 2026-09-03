//! The matrix operations, dispatched to the target that has the instructions for them.
//!
//! Nothing about a matrix fragment is shared between the two GPUs: AMD spreads a documented
//! layout across the wavefront and `CubeCL`'s lowering addresses it element by element, while
//! NVIDIA's WMMA fragment is opaque and only its own load and store instructions know where
//! anything sits. So unlike [`plane`](super::plane), where the targets agree on primitives and
//! differ only in the instructions, there is no shared body here worth writing -- each op is
//! forwarded whole.
//!
//! What is shared is that these ops exist in one place, so a target that does not implement
//! one says so rather than falling through to another target's instructions.

use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use pliron::input_err;
use thiserror::Error;

use crate::shared::to_llvm::prelude::*;
use crate::target::{CtxTarget, LlvmTarget};

/// A matrix operation on a target with no matrix instructions.
#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; a runtime that reaches it here has advertised \
     a matrix feature it cannot honour"
)]
pub struct MatrixOpUnsupported(LlvmTarget, &'static str);

/// The LLVM value a fragment of `matrix` lives in.
///
/// Both targets hold one in a vector, which is what lets the fragment be an `alloca` the other
/// ops load and store; what differs is how many elements of it a lane holds.
#[type_interface_impl]
impl CubeToLLVMType for MatrixType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        match ctx.target() {
            LlvmTarget::AmdGpu => crate::amdgpu::matrix::fragment_ty(ctx, self),
            LlvmTarget::Nvptx => crate::nvptx::matrix::fragment_ty(ctx, self),
            // Reached only if a CPU kernel declares a matrix, which needs a device advertising
            // a matrix feature; the CPU advertises none. A type conversion cannot report an
            // error, so this is the one place the refusal has to be a panic.
            LlvmTarget::Cpu => {
                unimplemented!("the CPU target has no matrix fragments")
            }
        }
    }
}

/// Forwards one matrix op to the target that has the instructions for it.
macro_rules! dispatch_matrix_op {
    ($cube_op:ty, $method:ident) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                operands_info: &OperandsInfo,
            ) -> Result<()> {
                match ctx.target() {
                    LlvmTarget::AmdGpu => {
                        crate::amdgpu::matrix::$method(self, ctx, rewriter, operands_info)
                    }
                    LlvmTarget::Nvptx => {
                        crate::nvptx::matrix::$method(self, ctx, rewriter, operands_info)
                    }
                    target => input_err!(
                        self.loc(ctx),
                        MatrixOpUnsupported(target, stringify!($cube_op))
                    ),
                }
            }
        }
    };
}

dispatch_matrix_op!(FillOp, fill);
dispatch_matrix_op!(LoadOp, load);
dispatch_matrix_op!(StoreOp, store);
dispatch_matrix_op!(MultiplyAccumulateOp, multiply_accumulate);
dispatch_matrix_op!(CastOp, cast);
dispatch_matrix_op!(RowIndexOp, row_index);
dispatch_matrix_op!(ColIndexOp, col_index);
dispatch_matrix_op!(MmaManualOp, mma_manual);
