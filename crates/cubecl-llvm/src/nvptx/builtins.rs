//! Resolves `cube.read_builtin` against the NVIDIA hardware.
//!
//! Where the CPU target emulates the launch grid with a loop nest
//! (see [`cpu::entrypoint`](crate::cpu::entrypoint)), the GPU *is* the grid. Every positional
//! builtin is one `llvm.nvvm.read.ptx.sreg.*` special register, `CubeCount*` included — which
//! is the one place NVIDIA is simpler than AMD, where the same answer has to be read out of
//! the HSA dispatch packet and divided by the cube dimension. So the whole pass is a
//! substitution with no control flow.

use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::general::ReadBuiltinOp;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::settings::Dim3;
use cubecl_core::ir::{Builtin, OpInserter, Scope};
use cubecl_core::prelude::*;
use pliron::builtin::ops::FuncOp;

use crate::cpu::entrypoint::{
    BuiltinValues, Replacer, absolute_pos, absolute_pos_x, absolute_pos_y, absolute_pos_z,
    constant, cube_count, cube_pos, set_dim_and_cluster_constants, unit_pos,
};
use crate::shared::intrinsic::{call_op, i32_ty};

/// The PTX special registers providing each positional builtin. `tid` is the unit within its
/// cube, `ctaid` the cube within the grid, `nctaid` the size of the grid in cubes.
const TID: [(&str, Builtin); 3] = [
    ("llvm.nvvm.read.ptx.sreg.tid.x", Builtin::UnitPosX),
    ("llvm.nvvm.read.ptx.sreg.tid.y", Builtin::UnitPosY),
    ("llvm.nvvm.read.ptx.sreg.tid.z", Builtin::UnitPosZ),
];

const CTAID: [(&str, Builtin); 3] = [
    ("llvm.nvvm.read.ptx.sreg.ctaid.x", Builtin::CubePosX),
    ("llvm.nvvm.read.ptx.sreg.ctaid.y", Builtin::CubePosY),
    ("llvm.nvvm.read.ptx.sreg.ctaid.z", Builtin::CubePosZ),
];

const NCTAID: [(&str, Builtin); 3] = [
    ("llvm.nvvm.read.ptx.sreg.nctaid.x", Builtin::CubeCountX),
    ("llvm.nvvm.read.ptx.sreg.nctaid.y", Builtin::CubeCountY),
    ("llvm.nvvm.read.ptx.sreg.nctaid.z", Builtin::CubeCountZ),
];

/// This lane's index within its warp.
const LANEID: &str = "llvm.nvvm.read.ptx.sreg.laneid";

/// Substitutes every `cube.read_builtin` with a special register read or a constant.
#[derive(Debug)]
pub struct InsertNvptxBuiltinsPass {
    /// Warp width of the target, which is 32 on every CUDA device. Carried rather than
    /// hardcoded so the value the builtin reports and the value the plane lowerings generate
    /// for are the same one.
    pub plane_dim: u32,
}

#[pass_name]
impl Pass for InsertNvptxBuiltinsPass {
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

        // Block start, not before the terminator: a branchless kernel's entry block ends in
        // `return`, so "before the terminator" puts these *after* the very `cube.read_builtin`
        // uses being substituted, leaving a use that references a later definition and failing
        // dominance. Everything computed here is self-contained, so nothing is lost by
        // computing it first. Same reasoning as the AMDGPU pass.
        let mut builtins = BuiltinValues::default();
        {
            let mut inserter = OpInserter::new_at_block_start(entry_block);
            let scope = Scope::from_context_and_inserter(ctx, &mut inserter);

            for (register, builtin) in TID.into_iter().chain(CTAID).chain(NCTAID) {
                builtins.set(builtin, read_sreg(&scope, register));
            }

            self.set_constants(&scope, &mut builtins, cube_dim, cluster_dim);
            derive_positions(&scope, &mut builtins, cube_dim);
        }

        let mut replacer = Replacer {
            builtins: &builtins,
            replacements: Vec::new(),
        };
        visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, &mut replacer, op, |ctx, replacer, op| {
            let builtin = op.builtin(ctx).0;
            let value = replacer.builtins.get(builtin).unwrap_or_else(|| {
                unimplemented!("the builtin {builtin:?} is not supported on the NVPTX target yet")
            });
            replacer.replacements.push((op.get_result(ctx), value));
        });
        for (old_value, new_value) in replacer.replacements {
            old_value.replace_all_uses_with(ctx, &new_value);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

impl InsertNvptxBuiltinsPass {
    /// The builtins fixed at launch, which the hardware would report from `ntid` but which are
    /// better as constants: they are known here, and constant-folding the index arithmetic
    /// built on them is most of what makes the derived positions cheap.
    fn set_constants(
        &self,
        scope: &Scope,
        builtins: &mut BuiltinValues,
        cube_dim: Dim3,
        cluster_dim: Dim3,
    ) {
        set_dim_and_cluster_constants(scope, builtins, cube_dim, cluster_dim);
        builtins.set(
            Builtin::PlaneDim,
            constant::expand(scope, self.plane_dim).value(scope),
        );
        builtins.set(Builtin::UnitPosPlane, read_sreg(scope, LANEID));
    }
}

/// Derived arithmetically from the hardware builtins, reusing the CPU target's `#[cube]`
/// helpers — the arithmetic relating a position to its components is the same everywhere, so
/// only where the components come from is a target's business.
fn derive_positions(scope: &Scope, builtins: &mut BuiltinValues, cube_dim: Dim3) {
    let unit_pos_x = builtins.expect(Builtin::UnitPosX);
    let unit_pos_y = builtins.expect(Builtin::UnitPosY);
    let unit_pos_z = builtins.expect(Builtin::UnitPosZ);
    let cube_pos_x = builtins.expect(Builtin::CubePosX);
    let cube_pos_y = builtins.expect(Builtin::CubePosY);
    let cube_pos_z = builtins.expect(Builtin::CubePosZ);
    let cube_count_x = builtins.expect(Builtin::CubeCountX);
    let cube_count_y = builtins.expect(Builtin::CubeCountY);
    let cube_count_z = builtins.expect(Builtin::CubeCountZ);

    let cube_count = cube_count::expand(
        scope,
        cube_count_x.into(),
        cube_count_y.into(),
        cube_count_z.into(),
    )
    .value(scope);
    builtins.set(Builtin::CubeCount, cube_count);

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

    let abs_x = absolute_pos_x::expand(scope, cube_pos_x.into(), unit_pos_x.into(), cube_dim.x)
        .value(scope);
    let abs_y = absolute_pos_y::expand(scope, cube_pos_y.into(), unit_pos_y.into(), cube_dim.y)
        .value(scope);
    let abs_z = absolute_pos_z::expand(scope, cube_pos_z.into(), unit_pos_z.into(), cube_dim.z)
        .value(scope);
    builtins.set(Builtin::AbsolutePosX, abs_x);
    builtins.set(Builtin::AbsolutePosY, abs_y);
    builtins.set(Builtin::AbsolutePosZ, abs_z);

    let absolute_pos = absolute_pos::expand(
        scope,
        abs_x.into(),
        abs_y.into(),
        abs_z.into(),
        cube_count_x.into(),
        cube_count_y.into(),
        cube_dim.x,
        cube_dim.y,
    )
    .value(scope);
    builtins.set(Builtin::AbsolutePos, absolute_pos);

    let cube_pos = cube_pos::expand(
        scope,
        cube_pos_x.into(),
        cube_pos_y.into(),
        cube_pos_z.into(),
        cube_count_x.into(),
        cube_count_y.into(),
    )
    .value(scope);
    builtins.set(Builtin::CubePos, cube_pos);
}

/// Reads the `i32` PTX special register `name`.
fn read_sreg(scope: &Scope, name: &str) -> Value {
    let ty = i32_ty(scope.ctx_mut());
    let op = call_op(scope.ctx_mut(), name, ty, vec![]);
    scope.register_with_result(&op)
}
