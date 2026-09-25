//! Builtins on the GPU targets: the ids the hardware hands a unit, and every other builtin
//! derived from them the same way on both.

use crate::{
    prelude::*,
    shared::builtin_values::{
        BuiltinValues, Replacer, absolute_pos, absolute_pos_x, absolute_pos_y, absolute_pos_z,
        constant, cube_count, cube_pos, plane_pos, set_dim_and_cluster_constants, unit_pos,
    },
};
use cubecl_core::{ir::dialect::general::ReadBuiltinOp, prelude::*};

/// What a target reads to place a unit in its launch, one value per axis.
pub struct LaunchValues {
    pub unit_pos: [Value; 3],
    pub cube_pos: [Value; 3],
    pub cube_count: [Value; 3],
    pub unit_pos_plane: Value,
}

/// Where a target reads its [`LaunchValues`] from.
pub trait ReadsLaunchValues: core::fmt::Debug {
    /// Reads them at `scope`, which is the start of the entry block, for a cube of `cube_dim`.
    fn read(&self, scope: &Scope, cube_dim: Dim3) -> LaunchValues;
}

/// Replaces every builtin read with the target's registers and what derives from them.
#[derive(Debug)]
pub struct InsertGpuBuiltinsPass {
    values: Box<dyn ReadsLaunchValues>,
    plane_dim: u32,
}

impl InsertGpuBuiltinsPass {
    /// `plane_dim` is the device's plane width.
    pub fn new(values: Box<dyn ReadsLaunchValues>, plane_dim: u32) -> Self {
        Self { values, plane_dim }
    }
}

impl LaunchValues {
    /// Sets the per-axis values and every builtin that combines them, for a cube of
    /// `cube_dim`.
    fn derive(&self, scope: &Scope, builtins: &mut BuiltinValues, cube_dim: Dim3) {
        let [unit_x, unit_y, unit_z] = self.unit_pos;
        let [cube_x, cube_y, cube_z] = self.cube_pos;
        let [count_x, count_y, count_z] = self.cube_count;

        let axes = [
            (Builtin::UnitPosX, unit_x),
            (Builtin::UnitPosY, unit_y),
            (Builtin::UnitPosZ, unit_z),
            (Builtin::CubePosX, cube_x),
            (Builtin::CubePosY, cube_y),
            (Builtin::CubePosZ, cube_z),
            (Builtin::CubeCountX, count_x),
            (Builtin::CubeCountY, count_y),
            (Builtin::CubeCountZ, count_z),
            (Builtin::UnitPosPlane, self.unit_pos_plane),
        ];
        for (builtin, value) in axes {
            builtins.set(builtin, value);
        }

        let count = cube_count::expand(scope, count_x.into(), count_y.into(), count_z.into());
        builtins.set(Builtin::CubeCount, count.value(scope));

        let unit = unit_pos::expand(
            scope,
            unit_x.into(),
            unit_y.into(),
            unit_z.into(),
            cube_dim.x,
            cube_dim.y,
        )
        .value(scope);
        builtins.set(Builtin::UnitPos, unit);

        let abs_x = absolute_pos_x::expand(scope, cube_x.into(), unit_x.into(), cube_dim.x);
        let abs_y = absolute_pos_y::expand(scope, cube_y.into(), unit_y.into(), cube_dim.y);
        let abs_z = absolute_pos_z::expand(scope, cube_z.into(), unit_z.into(), cube_dim.z);
        builtins.set(Builtin::AbsolutePosX, abs_x.value(scope));
        builtins.set(Builtin::AbsolutePosY, abs_y.value(scope));
        builtins.set(Builtin::AbsolutePosZ, abs_z.value(scope));

        let cube = cube_pos::expand(
            scope,
            cube_x.into(),
            cube_y.into(),
            cube_z.into(),
            count_x.into(),
            count_y.into(),
        )
        .value(scope);
        builtins.set(Builtin::CubePos, cube);

        let absolute = absolute_pos::expand(scope, cube.into(), unit.into(), cube_dim.num_elems());
        builtins.set(Builtin::AbsolutePos, absolute.value(scope));
    }
}

#[pass_name]
impl Pass for InsertGpuBuiltinsPass {
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

        // Builtin values must dominate their uses.
        let mut builtins = BuiltinValues::default();
        {
            let mut inserter = OpInserter::new_at_block_start(entry_block);
            let scope = Scope::from_context_and_inserter(ctx, &mut inserter);

            let values = self.values.read(&scope, cube_dim);
            set_dim_and_cluster_constants(&scope, &mut builtins, cube_dim, cluster_dim);
            builtins.set(
                Builtin::PlaneDim,
                constant::expand(&scope, self.plane_dim).value(&scope),
            );
            values.derive(&scope, &mut builtins, cube_dim);
            let plane = plane_pos::expand(
                &scope,
                builtins.expect(Builtin::UnitPos).into(),
                self.plane_dim,
            );
            builtins.set(Builtin::PlanePos, plane.value(&scope));
        }

        let mut replacer = Replacer {
            builtins: &builtins,
            replacements: Vec::new(),
        };
        visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, &mut replacer, op, |ctx, replacer, op| {
            let builtin = op.builtin(ctx).0;
            let value = replacer.builtins.get(builtin).unwrap_or_else(|| {
                unimplemented!("the builtin {builtin:?} is not supported on the GPU targets yet")
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
