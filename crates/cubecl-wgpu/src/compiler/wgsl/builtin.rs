use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    Builtin, CanMaterialize, ContextExt, FuncOpExt, Pure,
    attributes::{EntrypointInterface, FuncInterface, IndexAttr},
    dialect::general::ReadBuiltinOp,
    pliron::{
        builtin::{
            ops::FuncOp,
            types::{IntegerType, Signedness},
        },
        dict_key,
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op,
        },
        irbuild::match_rewrite::apply_match_rewrite,
    },
    prelude::*,
    settings::Dim3,
    types::VectorType,
};
use derive_more::{Display, From};
use hashbrown::HashMap;

use crate::compiler::wgsl::{KernelInfo, to_wgsl::wgsl_op_with_out};

#[pliron_attr(name = "wgsl.builtin", format = "$0", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, From)]
pub struct BuiltInAttr(pub BuiltIn);

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Display)]
#[display(rename_all = "snake_case")]
#[format]
pub enum BuiltIn {
    LocalInvocationIndex,
    LocalInvocationId,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SubgroupSize,
    SubgroupId,
    SubgroupInvocationId,
}

#[cube_op(name = "wgsl.read_scalar_builtin")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct ReadScalarBuiltin {
    value: Value,
}

wgsl_op_with_out!(ReadScalarBuiltin; |op, ctx| {
    format!("{}", op.value(ctx).name(ctx))
});

#[cube_op(name = "wgsl.read_dim3_builtin")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct ReadDim3Builtin {
    value: Value,
    dim: IndexAttr,
}

wgsl_op_with_out!(ReadDim3Builtin; |op, ctx| {
    format!("{}[{}]", op.value(ctx).name(ctx), op.dim(ctx).0)
});

pub struct LowerBuiltinsPass;

#[pass_name]
impl Pass for LowerBuiltinsPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let func = op.as_op::<FuncOp>(ctx).expect("Should run on func");

        // If we ever add dynamic function calls, this needs to become a call graph and update
        // each call with the args of nested functions. For now keep it simple.
        let op = func.get_operation();
        let mut rewriter = RewriteBuiltins {
            values: Default::default(),
            func,
        };
        res.ir_changed |= apply_match_rewrite(ctx, &mut rewriter, Default::default(), op)?;

        Ok(res)
    }
}

fn builtin_ty(ctx: &Context, builtin: BuiltIn) -> TypeHandle {
    let u32 = IntegerType::get(ctx, 32, Signedness::Signless).to_handle();
    let dim3 = VectorType::get(ctx, u32, 3).to_handle();
    match builtin {
        BuiltIn::NumWorkgroups => dim3,
        BuiltIn::WorkgroupId => dim3,
        BuiltIn::LocalInvocationId => dim3,
        BuiltIn::GlobalInvocationId => dim3,
        BuiltIn::LocalInvocationIndex => u32,
        BuiltIn::SubgroupSize => u32,
        BuiltIn::SubgroupId => u32,
        BuiltIn::SubgroupInvocationId => u32,
    }
}

dict_key!(ATTR_BUILTIN, "wgsl_builtin");

struct RewriteBuiltins {
    values: HashMap<BuiltIn, Value>,
    func: FuncOp,
}

impl RewriteBuiltins {
    fn get_value(&mut self, scope: &Scope, func: FuncOp, builtin: BuiltIn) -> Value {
        if let Some(&existing) = self.values.get(&builtin) {
            existing
        } else {
            let ctx = scope.ctx();
            let id = func.push_argument(ctx, builtin_ty(ctx, builtin));
            let value = func.get_entry_block(ctx).deref(ctx).get_argument(id);
            if func.get_entrypoint_abi(ctx).is_some() {
                func.set_arg_attr(ctx, id, &ATTR_BUILTIN, Box::new(BuiltInAttr(builtin)));
            }
            self.values.insert(builtin, value);
            value
        }
    }
}

impl MatchRewrite for RewriteBuiltins {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op.as_op::<ReadBuiltinOp>(ctx).unwrap();
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let ty = builtin.get_result(ctx).get_type(ctx);
        let builtin = builtin.builtin(ctx).0;
        let value = self.lower_builtin(&scope, ty, self.func, builtin);
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

#[cube]
fn absolute_pos() -> usize {
    cube_pos() * CUBE_DIM as usize + UNIT_POS as usize
}

#[cube]
fn cube_count() -> usize {
    CUBE_COUNT_X as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_Z as usize
}

#[cube]
fn cube_pos() -> usize {
    CUBE_POS_Z as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_X as usize
}

#[cube]
fn constant(#[comptime] value: u32) -> u32 {
    value
}

/// How a kernel's cube is laid out as a WGSL workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkgroupLayout {
    /// The cube's own `(x, y, z)` shape, read through the hardware's axis builtins.
    Shaped,
    /// One row of every unit in the cube. WGSL refuses `subgroup_id` and
    /// `subgroup_invocation_id` in a workgroup spanning `y` or `z`, so a kernel reading a unit's
    /// plane or lane runs flat and reads its axes back off its linear position, which is the
    /// same either way.
    Flat,
}

impl WorkgroupLayout {
    /// The layout of the kernel at `root`, run in cubes of `cube_dim`.
    pub fn new(ctx: &Context, root: Ptr<Operation>, cube_dim: &Dim3) -> Self {
        let spans_rows = cube_dim.y > 1 || cube_dim.z > 1;
        match spans_rows && reads_plane_ids(ctx, root) {
            true => Self::Flat,
            false => Self::Shaped,
        }
    }

    /// The `@workgroup_size` a cube of `cube_dim` launches with.
    pub fn workgroup_size(self, cube_dim: Dim3) -> Dim3 {
        match self {
            Self::Shaped => cube_dim,
            Self::Flat => Dim3::new_1d(cube_dim.num_elems()),
        }
    }
}

/// Whether the kernel at `root` reads a unit's plane or its lane in it, the builtins WGSL
/// only takes in a one-dimensional workgroup.
fn reads_plane_ids(ctx: &Context, root: Ptr<Operation>) -> bool {
    let mut reads = false;
    walk_op(
        ctx,
        &mut reads,
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, reads, node| {
            let IRNode::Operation(op) = node else {
                return;
            };
            *reads |= op.as_op::<ReadBuiltinOp>(ctx).is_some_and(|read| {
                matches!(
                    read.builtin(ctx).0,
                    Builtin::PlanePos | Builtin::UnitPosPlane
                )
            });
        },
    );
    reads
}

/// A unit's position along `x` in a flat cube: its linear position's lowest digit.
#[cube]
fn flat_unit_pos_x() -> u32 {
    UNIT_POS % CUBE_DIM_X
}

/// A unit's position along `y` in a flat cube: its linear position's middle digit.
#[cube]
fn flat_unit_pos_y() -> u32 {
    (UNIT_POS / CUBE_DIM_X) % CUBE_DIM_Y
}

/// A unit's position along `z` in a flat cube: its linear position's highest digit.
#[cube]
fn flat_unit_pos_z() -> u32 {
    UNIT_POS / (CUBE_DIM_X * CUBE_DIM_Y)
}

/// A unit's position in the whole kernel along `x` in a flat cube.
#[cube]
fn flat_absolute_pos_x() -> u32 {
    CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X
}

/// A unit's position in the whole kernel along `y` in a flat cube.
#[cube]
fn flat_absolute_pos_y() -> u32 {
    CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y
}

/// A unit's position in the whole kernel along `z` in a flat cube.
#[cube]
fn flat_absolute_pos_z() -> u32 {
    CUBE_POS_Z * CUBE_DIM_Z + UNIT_POS_Z
}

impl RewriteBuiltins {
    fn lower_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        func: FuncOp,
        builtin: Builtin,
    ) -> Value {
        let KernelInfo { cube_dim, layout } = *scope.ctx().aux_ty::<KernelInfo>();
        let flat = layout == WorkgroupLayout::Flat;
        match builtin {
            Builtin::UnitPos if cube_dim.num_elems() == 1 => {
                constant::expand(scope, 0).value(scope)
            }
            Builtin::UnitPosX if cube_dim.x == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosY if cube_dim.y == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosZ if cube_dim.z == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosX if flat => flat_unit_pos_x::expand(scope).value(scope),
            Builtin::UnitPosY if flat => flat_unit_pos_y::expand(scope).value(scope),
            Builtin::UnitPosZ if flat => flat_unit_pos_z::expand(scope).value(scope),
            Builtin::AbsolutePosX if flat => flat_absolute_pos_x::expand(scope).value(scope),
            Builtin::AbsolutePosY if flat => flat_absolute_pos_y::expand(scope).value(scope),
            Builtin::AbsolutePosZ if flat => flat_absolute_pos_z::expand(scope).value(scope),
            Builtin::UnitPos => {
                self.read_scalar_builtin(scope, ty, func, BuiltIn::LocalInvocationIndex)
            }
            Builtin::UnitPosX => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::LocalInvocationId, 0)
            }
            Builtin::UnitPosY => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::LocalInvocationId, 1)
            }
            Builtin::UnitPosZ => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::LocalInvocationId, 2)
            }
            Builtin::CubePosCluster => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterX => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterY => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterZ => constant::expand(scope, 0).value(scope),
            Builtin::CubePos => cube_pos::expand(scope).value(scope),
            Builtin::CubePosX => self.read_dim3_builtin(scope, ty, func, BuiltIn::WorkgroupId, 0),
            Builtin::CubePosY => self.read_dim3_builtin(scope, ty, func, BuiltIn::WorkgroupId, 1),
            Builtin::CubePosZ => self.read_dim3_builtin(scope, ty, func, BuiltIn::WorkgroupId, 2),
            Builtin::CubeDim => constant::expand(scope, cube_dim.num_elems()).value(scope),
            Builtin::CubeDimX => constant::expand(scope, cube_dim.x).value(scope),
            Builtin::CubeDimY => constant::expand(scope, cube_dim.y).value(scope),
            Builtin::CubeDimZ => constant::expand(scope, cube_dim.z).value(scope),
            Builtin::CubeClusterDim => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimX => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimY => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimZ => constant::expand(scope, 1).value(scope),
            Builtin::CubeCount => cube_count::expand(scope).value(scope),
            Builtin::CubeCountX => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::NumWorkgroups, 0)
            }
            Builtin::CubeCountY => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::NumWorkgroups, 1)
            }
            Builtin::CubeCountZ => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::NumWorkgroups, 2)
            }
            Builtin::PlaneDim => self.read_scalar_builtin(scope, ty, func, BuiltIn::SubgroupSize),
            Builtin::PlanePos => self.read_scalar_builtin(scope, ty, func, BuiltIn::SubgroupId),
            Builtin::UnitPosPlane => {
                self.read_scalar_builtin(scope, ty, func, BuiltIn::SubgroupInvocationId)
            }
            Builtin::AbsolutePos => absolute_pos::expand(scope).value(scope),
            Builtin::AbsolutePosX => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::GlobalInvocationId, 0)
            }
            Builtin::AbsolutePosY => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::GlobalInvocationId, 1)
            }
            Builtin::AbsolutePosZ => {
                self.read_dim3_builtin(scope, ty, func, BuiltIn::GlobalInvocationId, 2)
            }
        }
    }

    fn read_scalar_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        func: FuncOp,
        builtin: BuiltIn,
    ) -> Value {
        let value = self.get_value(scope, func, builtin);
        let ctx = scope.ctx_mut();
        let read = ReadScalarBuiltin::new(ctx, ty, value);
        scope.register_with_result(&read)
    }

    fn read_dim3_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        func: FuncOp,
        builtin: BuiltIn,
        dim: usize,
    ) -> Value {
        let value = self.get_value(scope, func, builtin);
        let ctx = scope.ctx_mut();
        let read = ReadDim3Builtin::new(ctx, ty, value, dim);
        scope.register_with_result(&read)
    }
}
