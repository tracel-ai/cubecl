use cubecl_core::ir::{Builtin, Scope, dialect::general::ReadBuiltinOp, prelude::*};
use pliron::context::Context;

use crate::{
    cuda::cuda_op_with_out,
    shared::{
        CompilationOptions, CompilationState,
        builtin::{LowerBuiltins, SharedBuiltin},
        signature::RequiresIncludesOp,
    },
    target::Cuda,
};

cuda_op_with_out!(ReadBuiltinOp, |op, ctx| {
    op.builtin(ctx).0.display_cuda(ctx)
});

#[op_interface_impl]
impl RequiresIncludesOp<Cuda> for ReadBuiltinOp {
    fn includes(&self, ctx: &Context) -> Vec<String> {
        match self.builtin(ctx).0 {
            Builtin::CubePosCluster
            | Builtin::CubePosClusterX
            | Builtin::CubePosClusterY
            | Builtin::CubePosClusterZ
            | Builtin::CubeClusterDim
            | Builtin::CubeClusterDimX
            | Builtin::CubeClusterDimY
            | Builtin::CubeClusterDimZ => vec!["cooperative_groups.h".into()],
            _ => vec![],
        }
    }
}

impl MatchRewrite for LowerBuiltins<Cuda> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op.as_op::<ReadBuiltinOp>(ctx).unwrap().builtin(ctx).0;
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        if let Some(new_value) = builtin.maybe_lower_shared(&scope) {
            rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        }
        Ok(())
    }
}

pub(crate) trait CudaBuiltin {
    fn display_cuda(&self, ctx: &Context) -> String;
}

impl CudaBuiltin for Builtin {
    fn display_cuda(&self, ctx: &Context) -> String {
        let clusters = ctx
            .aux_ty::<CompilationOptions>()
            .supports_features
            .clusters;
        let cluster_dim = ctx.aux_ty::<CompilationState>().cluster_dim;
        match self {
            Builtin::CubePosCluster if clusters => {
                "cooperative_groups::this_cluster().block_rank()".into()
            }
            Builtin::CubePosClusterX if clusters => {
                "cooperative_groups::this_cluster().block_index().x".into()
            }
            Builtin::CubePosClusterY if clusters => {
                "cooperative_groups::this_cluster().block_index().y".into()
            }
            Builtin::CubePosClusterZ if clusters => {
                "cooperative_groups::this_cluster().block_index().z".into()
            }
            Builtin::CubeClusterDim if clusters => format!("{}", cluster_dim.num_elems()),
            Builtin::CubeClusterDimX if clusters => format!("{}", cluster_dim.x),
            Builtin::CubeClusterDimY if clusters => format!("{}", cluster_dim.y),
            Builtin::CubeClusterDimZ if clusters => format!("{}", cluster_dim.z),
            _ => SharedBuiltin::display(self).into(),
        }
    }
}
