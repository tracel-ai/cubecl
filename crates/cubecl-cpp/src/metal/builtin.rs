use cubecl_core::{
    frontend::HasValue,
    ir::{Builtin, Scope, dialect::general::ReadBuiltinOp, prelude::*},
};
use pliron::{irbuild::match_rewrite::MatchRewrite, value::Value};

use crate::{
    metal::metal_op_with_out,
    shared::{
        CompilationState,
        builtin::{LowerBuiltins, absolute_pos, constant, cube_count, cube_pos},
    },
    target::Metal,
};

metal_op_with_out!(ReadBuiltinOp, |op, ctx| {
    op.builtin(ctx).0.display().into()
});

impl MatchRewrite for LowerBuiltins<Metal> {
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
        if let Some(new_value) = builtin.maybe_lower_metal(&scope) {
            rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        }
        Ok(())
    }
}

trait MetalBuiltin {
    fn display(&self) -> &'static str;
    fn maybe_lower_metal(&self, scope: &Scope) -> Option<Value>;
}

impl MetalBuiltin for Builtin {
    fn display(&self) -> &'static str {
        match self {
            Builtin::UnitPos => "thread_index_in_threadgroup",
            Builtin::UnitPosX => "thread_pos_in_threadgroup.x",
            Builtin::UnitPosY => "thread_pos_in_threadgroup.y",
            Builtin::UnitPosZ => "thread_pos_in_threadgroup.z",
            Builtin::CubePosX => "threadgroup_pos_in_grid.x",
            Builtin::CubePosY => "threadgroup_pos_in_grid.y",
            Builtin::CubePosZ => "threadgroup_pos_in_grid.z",
            Builtin::CubeDimX => "threads_per_threadgroup.x",
            Builtin::CubeDimY => "threads_per_threadgroup.y",
            Builtin::CubeDimZ => "threads_per_threadgroup.z",
            Builtin::CubeCountX => "threadgroups_per_grid.x",
            Builtin::CubeCountY => "threadgroups_per_grid.y",
            Builtin::CubeCountZ => "threadgroups_per_grid.z",
            Builtin::PlaneDim => "simd_size",
            Builtin::PlanePos => "simd_group_id",
            Builtin::UnitPosPlane => "simd_lane_id",
            Builtin::AbsolutePosX => "thread_pos_in_grid.x",
            Builtin::AbsolutePosY => "thread_pos_in_grid.y",
            Builtin::AbsolutePosZ => "thread_pos_in_grid.z",
            _ => unreachable!("Should be lowered"),
        }
    }

    fn maybe_lower_metal(&self, scope: &Scope) -> Option<Value> {
        let cube_dim = scope.ctx().aux_ty::<CompilationState>().cube_dim;
        match self {
            Builtin::UnitPos => None,
            // This is common enough to be worth replacing. Z is almost always 1, and Y is often 1.
            // Replacing it with a constant allows simplifying the positional math
            Builtin::UnitPosX if cube_dim.x == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosY if cube_dim.y == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosZ if cube_dim.z == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosX | Builtin::UnitPosY | Builtin::UnitPosZ => None,
            Builtin::CubePosCluster => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterX => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterY => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterZ => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePos => Some(cube_pos::expand(scope).value(scope)),
            Builtin::CubePosX | Builtin::CubePosY | Builtin::CubePosZ => None,
            Builtin::CubeDim => Some(constant::expand(scope, cube_dim.num_elems()).value(scope)),
            Builtin::CubeDimX => Some(constant::expand(scope, cube_dim.x).value(scope)),
            Builtin::CubeDimY => Some(constant::expand(scope, cube_dim.y).value(scope)),
            Builtin::CubeDimZ => Some(constant::expand(scope, cube_dim.z).value(scope)),
            Builtin::CubeClusterDim => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimX => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimY => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimZ => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeCount => Some(cube_count::expand(scope).value(scope)),
            Builtin::CubeCountX | Builtin::CubeCountY | Builtin::CubeCountZ => None,
            Builtin::PlaneDim => None,
            Builtin::PlanePos => None,
            Builtin::UnitPosPlane => None,
            Builtin::AbsolutePos => Some(absolute_pos::expand(scope).value(scope)),
            Builtin::AbsolutePosX | Builtin::AbsolutePosY | Builtin::AbsolutePosZ => None,
        }
    }
}
