use cubecl_core::{
    frontend::HasValue,
    ir::{Builtin, Scope, dialect::general::ReadBuiltinOp, prelude::*},
};
use pliron::irbuild::match_rewrite::MatchRewrite;

use crate::{
    error::{CompileError, EmissionErrors},
    opencl::{OpenClOptions, opencl_op_with_out},
    shared::{
        CompilationState,
        builtin::{LowerBuiltins, absolute_pos, constant, cube_count, cube_pos},
    },
    target::OpenCl,
};

impl MatchRewrite for LowerBuiltins<OpenCl> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op
            .as_op::<ReadBuiltinOp>(ctx)
            .expect("the rewrite should only see the op it matched")
            .builtin(ctx)
            .0;
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        if let Some(new_value) = lower(builtin, &scope) {
            rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        }
        Ok(())
    }
}

/// The builtins composed from others or known at compile time; the rest are work-item functions.
fn lower(builtin: Builtin, scope: &Scope) -> Option<Value> {
    let cube_dim = scope.ctx().aux_ty::<CompilationState>().cube_dim;
    let constant = |value| Some(constant::expand(scope, value).value(scope));
    match builtin {
        Builtin::UnitPosX if cube_dim.x == 1 => constant(0),
        Builtin::UnitPosY if cube_dim.y == 1 => constant(0),
        Builtin::UnitPosZ if cube_dim.z == 1 => constant(0),
        Builtin::CubePosCluster
        | Builtin::CubePosClusterX
        | Builtin::CubePosClusterY
        | Builtin::CubePosClusterZ => constant(0),
        Builtin::CubeDim => constant(cube_dim.num_elems()),
        Builtin::CubeDimX => constant(cube_dim.x),
        Builtin::CubeDimY => constant(cube_dim.y),
        Builtin::CubeDimZ => constant(cube_dim.z),
        Builtin::CubeClusterDim
        | Builtin::CubeClusterDimX
        | Builtin::CubeClusterDimY
        | Builtin::CubeClusterDimZ => constant(1),
        Builtin::CubePos => Some(cube_pos::expand(scope).value(scope)),
        Builtin::CubeCount => Some(cube_count::expand(scope).value(scope)),
        Builtin::AbsolutePos => Some(absolute_pos::expand(scope).value(scope)),
        _ => None,
    }
}

opencl_op_with_out!(ReadBuiltinOp, |op, ctx| {
    let ty = op.get_result(ctx).get_type(ctx);
    let ty = crate::shared::ty::TypeExtCPP::to_cpp(&ty, ctx);
    let builtin = op.builtin(ctx).0;
    let plane = matches!(
        builtin,
        Builtin::PlaneDim | Builtin::PlanePos | Builtin::UnitPosPlane
    );
    if plane && OpenClOptions::of(ctx).subgroup_size.is_none() {
        // The device has no planes (or not every plane operation): their indices are meaningless.
        let error = CompileError::UnsupportedOp(format!("{builtin:?} without planes"));
        ctx.aux_ty::<EmissionErrors>().record(error);
    }
    let call = match builtin {
        Builtin::UnitPos => "get_local_linear_id()",
        Builtin::UnitPosX => "get_local_id(0)",
        Builtin::UnitPosY => "get_local_id(1)",
        Builtin::UnitPosZ => "get_local_id(2)",
        Builtin::CubePosX => "get_group_id(0)",
        Builtin::CubePosY => "get_group_id(1)",
        Builtin::CubePosZ => "get_group_id(2)",
        Builtin::CubeCountX => "get_num_groups(0)",
        Builtin::CubeCountY => "get_num_groups(1)",
        Builtin::CubeCountZ => "get_num_groups(2)",
        Builtin::AbsolutePosX => "get_global_id(0)",
        Builtin::AbsolutePosY => "get_global_id(1)",
        Builtin::AbsolutePosZ => "get_global_id(2)",
        Builtin::PlaneDim => "get_max_sub_group_size()",
        Builtin::PlanePos => "get_sub_group_id()",
        Builtin::UnitPosPlane => "get_sub_group_local_id()",
        other => unreachable!("{other:?} should be lowered"),
    };
    format!("({ty}){call}")
});
