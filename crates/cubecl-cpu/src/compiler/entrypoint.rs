use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::branch::{RangeLoopOp, YieldOp};
use cubecl_core::ir::dialect::general::ReadBuiltinOp;
use cubecl_core::ir::dialect::memory::LoadOp;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::settings::Dim3;
use cubecl_core::ir::{Builtin, OpInserter, Scope};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use pliron::basic_block::BasicBlock;
use pliron::builtin::ops::FuncOp;
use pliron::linked_list::ContainsLinkedList;

pub const CPU_RUNTIME_BUILTINS: [Builtin; 6] = [
    Builtin::CubeCountX,
    Builtin::CubeCountY,
    Builtin::CubeCountZ,
    Builtin::UnitPosX,
    Builtin::UnitPosY,
    Builtin::UnitPosZ,
];

const NB_BUILTIN: usize = 31;

/// Every builtin the skeleton knows how to provide, whether it comes from a function argument, a
/// compile time constant, or a value computed by the emulation loop.
#[derive(Default)]
struct BuiltinValues([Option<Value>; NB_BUILTIN]);

impl BuiltinValues {
    fn set(&mut self, builtin: Builtin, value: Value) {
        self.0[builtin as usize] = Some(value);
    }

    fn get(&self, builtin: Builtin) -> Option<Value> {
        self.0[builtin as usize]
    }

    fn expect(&self, builtin: Builtin) -> Value {
        self.get(builtin)
            .unwrap_or_else(|| panic!("Builtin {builtin:?} should have been computed already"))
    }
}

/// Pending `cube.read_builtin` replacements, gathered during the IR walk so they can be applied
/// afterwards, once the walker no longer holds the ops borrowed.
struct Replacer<'a> {
    builtins: &'a BuiltinValues,
    replacements: Vec<(Value, Value)>,
}

#[cube]
fn constant(#[comptime] value: u32) -> u32 {
    value
}

#[cube]
fn unit_pos(
    unit_pos_x: u32,
    unit_pos_y: u32,
    unit_pos_z: u32,
    #[comptime] cube_dim_x: u32,
    #[comptime] cube_dim_y: u32,
) -> u32 {
    unit_pos_x + unit_pos_y * cube_dim_x + unit_pos_z * cube_dim_x * cube_dim_y
}

#[cube]
fn absolute_pos_x(cube_pos_x: u32, unit_pos_x: u32, #[comptime] cube_dim_x: u32) -> u32 {
    cube_pos_x * cube_dim_x + unit_pos_x
}

#[cube]
fn absolute_pos_y(cube_pos_y: u32, unit_pos_y: u32, #[comptime] cube_dim_y: u32) -> u32 {
    cube_pos_y * cube_dim_y + unit_pos_y
}

#[cube]
fn absolute_pos_z(cube_pos_z: u32, unit_pos_z: u32, #[comptime] cube_dim_z: u32) -> u32 {
    cube_pos_z * cube_dim_z + unit_pos_z
}

#[cube]
fn absolute_pos(
    absolute_pos_x: u32,
    absolute_pos_y: u32,
    absolute_pos_z: u32,
    cube_count_x: u32,
    cube_count_y: u32,
    #[comptime] cube_dim_x: u32,
    #[comptime] cube_dim_y: u32,
) -> usize {
    let units_x = cube_count_x as usize * cube_dim_x as usize;
    let units_y = cube_count_y as usize * cube_dim_y as usize;
    absolute_pos_z as usize * units_x * units_y
        + absolute_pos_y as usize * units_x
        + absolute_pos_x as usize
}

#[cube]
fn cube_pos(
    cube_pos_x: u32,
    cube_pos_y: u32,
    cube_pos_z: u32,
    cube_count_x: u32,
    cube_count_y: u32,
) -> usize {
    cube_pos_z as usize * cube_count_x as usize * cube_count_y as usize
        + cube_pos_y as usize * cube_count_x as usize
        + cube_pos_x as usize
}

#[cube]
fn cube_count(cube_count_x: u32, cube_count_y: u32, cube_count_z: u32) -> usize {
    cube_count_x as usize * cube_count_y as usize * cube_count_z as usize
}

/// Emulates the launch grid on the CPU.
///
/// The jitted kernel is invoked once per unit of a cube, and walks the whole cube grid itself
/// through a `cube_pos_z / cube_pos_y / cube_pos_x` loop nest. The cube count and the unit position
/// come from the host as arguments (see [`CPU_RUNTIME_BUILTINS`]), while the cube dim is compiled in as
/// constants, so the positional math folds away in the constant propagation pass.
#[derive(Default)]
pub struct InsertConstantEmulationPass;

#[pass_name]
impl Pass for InsertConstantEmulationPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };
        let Some(abi) = func.get_entrypoint_abi(ctx) else {
            return Ok(res);
        };
        let cube_dim = abi.cube_dim;
        let cluster_dim = abi.cluster_dim.unwrap_or(Dim3::new_single());

        let entry_block = func.get_entry_block(ctx);
        let terminator = entry_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("Entry block should be terminated with a return");

        let body_ops: Vec<Ptr<Operation>> = entry_block
            .deref(ctx)
            .iter(ctx)
            .filter(|body_op| *body_op != terminator)
            .collect();
        for body_op in body_ops.iter() {
            body_op.unlink(ctx);
        }

        let mut builtins = BuiltinValues::default();

        let body_block = {
            let mut inserter = OpInserter::new_before_operation(terminator);
            let scope = Scope::from_context_and_inserter(ctx, &mut inserter);

            let u32_ty = u32::__expand_as_type(&scope);
            for builtin in CPU_RUNTIME_BUILTINS {
                let arg = func.push_argument(scope.ctx(), u32_ty);
                let value = entry_block.deref(scope.ctx()).get_argument(arg);
                builtins.set(builtin, value);
            }

            insert_skeleton(&scope, &mut builtins, cube_dim, cluster_dim)
        };

        for body_op in body_ops {
            body_op.insert_at_back(body_block, ctx);
        }
        YieldOp::new(ctx)
            .get_operation()
            .insert_at_back(body_block, ctx);

        // The walker keeps the visited op borrowed, so we can't replace uses in place. Collect the
        // replacements first, then apply them once the walk is done.
        let mut replacer = Replacer {
            builtins: &builtins,
            replacements: Vec::new(),
        };
        visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, &mut replacer, op, |ctx, replacer, op| {
            if let Some(value) = replacer.builtins.get(op.builtin(ctx).0) {
                replacer.replacements.push((op.get_result(ctx), value));
            }
        });
        for (old_value, new_value) in replacer.replacements {
            old_value.replace_all_uses_with(ctx, &new_value);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

fn insert_skeleton(
    scope: &Scope,
    builtins: &mut BuiltinValues,
    cube_dim: Dim3,
    cluster_dim: Dim3,
) -> Ptr<BasicBlock> {
    let mut set_const = |builtin: Builtin, value: u32| {
        builtins.set(builtin, constant::expand(scope, value).value(scope));
    };

    set_const(Builtin::CubeDimX, cube_dim.x);
    set_const(Builtin::CubeDimY, cube_dim.y);
    set_const(Builtin::CubeDimZ, cube_dim.z);
    set_const(Builtin::CubeDim, cube_dim.num_elems());

    set_const(Builtin::CubeClusterDimX, cluster_dim.x);
    set_const(Builtin::CubeClusterDimY, cluster_dim.y);
    set_const(Builtin::CubeClusterDimZ, cluster_dim.z);
    set_const(Builtin::CubeClusterDim, cluster_dim.num_elems());

    // A CPU has no cluster, so a cube always sits at position 0 of its cluster.
    set_const(Builtin::CubePosCluster, 0);
    set_const(Builtin::CubePosClusterX, 0);
    set_const(Builtin::CubePosClusterY, 0);
    set_const(Builtin::CubePosClusterZ, 0);

    // A unit is a scalar thread on the CPU, so a plane holds exactly one unit.
    set_const(Builtin::PlaneDim, 1);
    set_const(Builtin::UnitPosPlane, 0);

    let zero = constant::expand(scope, 0).value(scope);
    let one = constant::expand(scope, 1).value(scope);

    let unit_pos_x = builtins.expect(Builtin::UnitPosX);
    let unit_pos_y = builtins.expect(Builtin::UnitPosY);
    let unit_pos_z = builtins.expect(Builtin::UnitPosZ);
    let cube_count_x = builtins.expect(Builtin::CubeCountX);
    let cube_count_y = builtins.expect(Builtin::CubeCountY);
    let cube_count_z = builtins.expect(Builtin::CubeCountZ);

    let unit_pos = unit_pos::expand(
        scope,
        unit_pos_x.into(),
        unit_pos_y.into(),
        unit_pos_z.into(),
        cube_dim.x,
        cube_dim.y,
    )
    .value(scope);
    builtins.set(Builtin::UnitPos, unit_pos);
    builtins.set(Builtin::PlanePos, unit_pos);

    let cube_count = cube_count::expand(
        scope,
        cube_count_x.into(),
        cube_count_y.into(),
        cube_count_z.into(),
    )
    .value(scope);
    builtins.set(Builtin::CubeCount, cube_count);

    let u32_ty = u32::__expand_as_type(scope);

    let iter_z = scope.create_local_mut(u32_ty, None);
    let loop_z = RangeLoopOp::new(scope.ctx_mut(), iter_z, zero, cube_count_z, one);
    let scope_z = scope.child(OpInserter::new_at_block_end(loop_z.loop_body(scope.ctx())));
    {
        let load = LoadOp::new(scope_z.ctx_mut(), iter_z);
        let cube_pos_z = scope_z.register_with_result(&load);
        builtins.set(Builtin::CubePosZ, cube_pos_z);

        let absolute_pos_z =
            absolute_pos_z::expand(&scope_z, cube_pos_z.into(), unit_pos_z.into(), cube_dim.z)
                .value(&scope_z);
        builtins.set(Builtin::AbsolutePosZ, absolute_pos_z);
    }

    let iter_y = scope_z.create_local_mut(u32_ty, None);
    let loop_y = RangeLoopOp::new(scope_z.ctx_mut(), iter_y, zero, cube_count_y, one);
    let scope_y = scope_z.child(OpInserter::new_at_block_end(
        loop_y.loop_body(scope_z.ctx()),
    ));
    {
        let load = LoadOp::new(scope_y.ctx_mut(), iter_y);
        let cube_pos_y = scope_y.register_with_result(&load);
        builtins.set(Builtin::CubePosY, cube_pos_y);

        let absolute_pos_y =
            absolute_pos_y::expand(&scope_y, cube_pos_y.into(), unit_pos_y.into(), cube_dim.y)
                .value(&scope_y);
        builtins.set(Builtin::AbsolutePosY, absolute_pos_y);
    }

    let iter_x = scope_y.create_local_mut(u32_ty, None);
    let loop_x = RangeLoopOp::new(scope_y.ctx_mut(), iter_x, zero, cube_count_x, one);
    let body_block = loop_x.loop_body(scope_y.ctx());
    let scope_x = scope_y.child(OpInserter::new_at_block_end(body_block));
    {
        let load = LoadOp::new(scope_x.ctx_mut(), iter_x);
        let cube_pos_x = scope_x.register_with_result(&load);
        builtins.set(Builtin::CubePosX, cube_pos_x);

        let absolute_pos_x =
            absolute_pos_x::expand(&scope_x, cube_pos_x.into(), unit_pos_x.into(), cube_dim.x)
                .value(&scope_x);
        builtins.set(Builtin::AbsolutePosX, absolute_pos_x);

        let absolute_pos = absolute_pos::expand(
            &scope_x,
            absolute_pos_x.into(),
            builtins.expect(Builtin::AbsolutePosY).into(),
            builtins.expect(Builtin::AbsolutePosZ).into(),
            cube_count_x.into(),
            cube_count_y.into(),
            cube_dim.x,
            cube_dim.y,
        )
        .value(&scope_x);
        builtins.set(Builtin::AbsolutePos, absolute_pos);

        let cube_pos = cube_pos::expand(
            &scope_x,
            cube_pos_x.into(),
            builtins.expect(Builtin::CubePosY).into(),
            builtins.expect(Builtin::CubePosZ).into(),
            cube_count_x.into(),
            cube_count_y.into(),
        )
        .value(&scope_x);
        builtins.set(Builtin::CubePos, cube_pos);
    }

    scope_y.register(&loop_x);
    scope_y.terminate_yield();
    scope_z.register(&loop_y);
    scope_z.terminate_yield();
    scope.register(&loop_z);

    body_block
}
