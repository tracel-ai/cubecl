use core::{any::type_name, marker::PhantomData};

use cubecl_core::{
    self as cubecl,
    ir::{
        Builtin, ContextExt,
        dialect::general::ReadBuiltinOp,
        prelude::{AnalysisManager, Context, Operation, PassResult, Ptr, Result},
    },
    prelude::*,
};
use pliron::{
    irbuild::{
        IRStatus,
        match_rewrite::{MatchRewrite, apply_match_rewrite},
    },
    pass_manager::Pass,
    value::Value,
};

use crate::shared::{CompilationState, shared_op_with_out};

shared_op_with_out!(ReadBuiltinOp, |op, ctx| {
    op.builtin(ctx).0.display().into()
});

#[cube]
pub fn absolute_pos() -> usize {
    let cubes_x = CUBE_COUNT_X as usize;
    let cubes_y = CUBE_COUNT_Y as usize;
    let cube_dim_x = CUBE_DIM_X as usize;
    let cube_dim_y = CUBE_DIM_Y as usize;
    let z = ABSOLUTE_POS_Z as usize * cubes_x * cube_dim_x * cubes_y * cube_dim_y;
    let y = ABSOLUTE_POS_Y as usize * cubes_x * cube_dim_x;
    z + y + ABSOLUTE_POS_X as usize
}

#[cube]
pub fn absolute_pos_x() -> u32 {
    CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X
}

#[cube]
pub fn absolute_pos_y() -> u32 {
    CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y
}

#[cube]
pub fn absolute_pos_z() -> u32 {
    CUBE_POS_Z * CUBE_DIM_Z + UNIT_POS_Z
}

#[cube]
pub fn cube_count() -> usize {
    CUBE_COUNT_X as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_Z as usize
}

#[cube]
pub fn cube_pos() -> usize {
    CUBE_POS_Z as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_X as usize
}

#[cube]
pub fn unit_pos() -> u32 {
    UNIT_POS_X + UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_Z * CUBE_DIM_X * CUBE_DIM_Y
}

#[cube]
pub fn unit_pos_plane() -> u32 {
    UNIT_POS % PLANE_DIM
}

#[cube]
pub fn plane_pos() -> u32 {
    UNIT_POS / PLANE_DIM
}

#[cube]
pub fn constant(#[comptime] value: u32) -> u32 {
    value
}

#[derive(Default)]
pub struct LowerBuiltins<T> {
    _target: PhantomData<T>,
}

#[derive(Default)]
pub struct LowerBuiltinsPass<T> {
    _ty: PhantomData<T>,
}

impl<T: Default> Pass for LowerBuiltinsPass<T>
where
    LowerBuiltins<T>: MatchRewrite,
{
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        while apply_match_rewrite(ctx, LowerBuiltins::<T>::default(), op)? == IRStatus::Changed {
            res.ir_changed |= IRStatus::Changed
        }
        Ok(res)
    }
}

pub(crate) trait SharedBuiltin {
    fn display(&self) -> &'static str;
    fn maybe_lower_shared(&self, scope: &Scope) -> Option<Value>;
}

impl SharedBuiltin for Builtin {
    fn display(&self) -> &'static str {
        match self {
            Builtin::UnitPosX => "threadIdx.x",
            Builtin::UnitPosY => "threadIdx.y",
            Builtin::UnitPosZ => "threadIdx.z",
            Builtin::CubePosCluster => "0",
            Builtin::CubePosClusterX | Builtin::CubePosClusterY | Builtin::CubePosClusterZ => "0",
            Builtin::CubePosX => "blockIdx.x",
            Builtin::CubePosY => "blockIdx.y",
            Builtin::CubePosZ => "blockIdx.z",
            Builtin::CubeDimX => "blockDim.x",
            Builtin::CubeDimY => "blockDim.y",
            Builtin::CubeDimZ => "blockDim.z",
            Builtin::CubeCountX => "gridDim.x",
            Builtin::CubeCountY => "gridDim.y",
            Builtin::CubeCountZ => "gridDim.z",
            Builtin::PlaneDim => "warpSize",
            _ => unreachable!("Should be lowered"),
        }
    }

    fn maybe_lower_shared(&self, scope: &Scope) -> Option<Value> {
        let cube_dim = scope.ctx().aux_ty::<CompilationState>().cube_dim;
        let cluster = scope.ctx().aux_ty::<CompilationState>().cluster_dim;
        match self {
            Builtin::UnitPos => Some(unit_pos::expand(scope).value(scope)),
            Builtin::UnitPosX | Builtin::UnitPosY | Builtin::UnitPosZ => None,
            Builtin::CubePosCluster => None,
            Builtin::CubePosClusterX | Builtin::CubePosClusterY | Builtin::CubePosClusterZ => None,
            Builtin::CubePos => Some(cube_pos::expand(scope).value(scope)),
            Builtin::CubePosX | Builtin::CubePosY | Builtin::CubePosZ => None,
            Builtin::CubeDim => Some(constant::expand(scope, cube_dim.num_elems()).value(scope)),
            Builtin::CubeDimX => Some(constant::expand(scope, cube_dim.x).value(scope)),
            Builtin::CubeDimY => Some(constant::expand(scope, cube_dim.y).value(scope)),
            Builtin::CubeDimZ => Some(constant::expand(scope, cube_dim.z).value(scope)),
            Builtin::CubeClusterDim => {
                Some(constant::expand(scope, cluster.num_elems()).value(scope))
            }
            Builtin::CubeClusterDimX => Some(constant::expand(scope, cluster.x).value(scope)),
            Builtin::CubeClusterDimY => Some(constant::expand(scope, cluster.y).value(scope)),
            Builtin::CubeClusterDimZ => Some(constant::expand(scope, cluster.z).value(scope)),
            Builtin::CubeCount => Some(cube_count::expand(scope).value(scope)),
            Builtin::CubeCountX | Builtin::CubeCountY | Builtin::CubeCountZ => None,
            Builtin::PlaneDim => None,
            Builtin::PlanePos => Some(plane_pos::expand(scope).value(scope)),
            Builtin::UnitPosPlane => Some(unit_pos_plane::expand(scope).value(scope)),
            Builtin::AbsolutePos => Some(absolute_pos::expand(scope).value(scope)),
            Builtin::AbsolutePosX => Some(absolute_pos_x::expand(scope).value(scope)),
            Builtin::AbsolutePosY => Some(absolute_pos_y::expand(scope).value(scope)),
            Builtin::AbsolutePosZ => Some(absolute_pos_z::expand(scope).value(scope)),
        }
    }
}
