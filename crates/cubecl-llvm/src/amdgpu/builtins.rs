//! Resolves `cube.read_builtin` against the AMDGPU hardware.
//!
//! Where the CPU target emulates the launch grid with a loop nest
//! (see `shared::entrypoint::InsertConstantEmulationPass`), the GPU *is* the
//! grid: the positional builtins come from `llvm.amdgcn` intrinsics, `CubeCount*`
//! comes from the HSA kernel dispatch packet, and the dimensional builtins are
//! compile-time constants, so the whole pass is a substitution with no control flow.

use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::general::ReadBuiltinOp;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::settings::Dim3;
use cubecl_core::ir::{Builtin, OpInserter, Scope};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{IntegerType, Signedness};

use crate::amdgpu::intrinsic::{call_op, i32_ty, lane_id_ops};
use pliron_llvm::ops::{GepIndex, GetElementPtrOp, LoadOp};
use pliron_llvm::types::PointerType as LlvmPointerType;

use crate::cpu::entrypoint::{
    BuiltinValues, Replacer, absolute_pos, absolute_pos_x, absolute_pos_y, absolute_pos_z,
    constant, cube_count, cube_pos, set_dim_and_cluster_constants, unit_pos,
};

/// The `llvm.amdgcn` intrinsics providing each positional builtin.
const WORKITEM_ID: [(&str, Builtin); 3] = [
    ("llvm.amdgcn.workitem.id.x", Builtin::UnitPosX),
    ("llvm.amdgcn.workitem.id.y", Builtin::UnitPosY),
    ("llvm.amdgcn.workitem.id.z", Builtin::UnitPosZ),
];

const WORKGROUP_ID: [(&str, Builtin); 3] = [
    ("llvm.amdgcn.workgroup.id.x", Builtin::CubePosX),
    ("llvm.amdgcn.workgroup.id.y", Builtin::CubePosY),
    ("llvm.amdgcn.workgroup.id.z", Builtin::CubePosZ),
];

/// AMDGPU's constant address space. This is where the HSA runtime maps the kernel dispatch
/// packet that `llvm.amdgcn.dispatch.ptr` points to.
const CONSTANT_ADDRESS_SPACE: u32 = 4;

/// Byte offsets of the `grid_size_{x,y,z}` fields within `has_kernel_dispatch_packet_t`
/// (verified against `rocm-runtime`'s `hsa/hsa.h`). These fields count work-items launched
/// along each axis, not workgroups.
const GRID_SIZE_X_OFFSET: u32 = 12;
const GRID_SIZE_Y_OFFSET: u32 = 16;
const GRID_SIZE_Z_OFFSET: u32 = 20;

/// Substitutes every `cube.read_builtin` with an intrinsic call or a constant.
#[derive(Debug)]
pub struct InsertAmdgpuBuiltinsPass {
    /// Wavefront width of the target: 32 on RDNA, 64 on CDNA.
    pub plane_dim: u32,
}

#[pass_name]
impl Pass for InsertAmdgpuBuiltinsPass {
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

        // Block start, not before the terminator: a branchless kernel's entry block
        // ends in `return`, so "before the terminator" puts these *after* the very
        // `cube.read_builtin` uses being substituted, leaving a use that references a
        // later definition and failing dominance. Everything computed here is
        // self-contained, so nothing is lost by computing it first.
        let mut builtins = BuiltinValues::default();
        {
            let mut inserter = OpInserter::new_at_block_start(entry_block);
            let scope = Scope::from_context_and_inserter(ctx, &mut inserter);

            for (intrinsic, builtin) in WORKITEM_ID.into_iter().chain(WORKGROUP_ID) {
                builtins.set(builtin, call_i32_intrinsic(&scope, intrinsic));
            }

            self.set_constants(&scope, &mut builtins, cube_dim, cluster_dim);
            set_cube_count(&scope, &mut builtins, cube_dim);
            derive_positions(&scope, &mut builtins, cube_dim);
        }

        let mut replacer = Replacer {
            builtins: &builtins,
            replacements: Vec::new(),
        };
        visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, &mut replacer, op, |ctx, replacer, op| {
            let builtin = op.builtin(ctx).0;
            let value = replacer.builtins.get(builtin).unwrap_or_else(|| {
                unimplemented!("the builtin {builtin:?} is not supported on the AMDGPU target yet")
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

impl InsertAmdgpuBuiltinsPass {
    /// The builtins the hardware does not report because they are fixed at launch.
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
        builtins.set(Builtin::UnitPosPlane, unit_pos_plane(scope));
    }
}

/// `CubeCount*` has no intrinsic; it comes from the dispatch packet via
/// `llvm.amdgcn.dispatch.ptr`. `grid_size_*` counts work-items, not workgroups, so
/// `cube_count = grid_size / cube_dim` per axis.
fn set_cube_count(scope: &Scope, builtins: &mut BuiltinValues, cube_dim: Dim3) {
    let dispatch_ptr = dispatch_ptr(scope);
    let grid_size_x = load_u32_at(scope, dispatch_ptr, GRID_SIZE_X_OFFSET);
    let grid_size_y = load_u32_at(scope, dispatch_ptr, GRID_SIZE_Y_OFFSET);
    let grid_size_z = load_u32_at(scope, dispatch_ptr, GRID_SIZE_Z_OFFSET);

    let cube_count_x =
        cube_count_component::expand(scope, grid_size_x.into(), cube_dim.x).value(scope);
    let cube_count_y =
        cube_count_component::expand(scope, grid_size_y.into(), cube_dim.y).value(scope);
    let cube_count_z =
        cube_count_component::expand(scope, grid_size_z.into(), cube_dim.z).value(scope);
    builtins.set(Builtin::CubeCountX, cube_count_x);
    builtins.set(Builtin::CubeCountY, cube_count_y);
    builtins.set(Builtin::CubeCountZ, cube_count_z);

    let cube_count = cube_count::expand(
        scope,
        cube_count_x.into(),
        cube_count_y.into(),
        cube_count_z.into(),
    )
    .value(scope);
    builtins.set(Builtin::CubeCount, cube_count);
}

/// `cube_count = grid_size / cube_dim`, per axis. `cube_dim` is comptime, so this constant-folds.
#[cube]
fn cube_count_component(grid_size: u32, #[comptime] cube_dim: u32) -> u32 {
    grid_size / cube_dim
}

/// Derived arithmetically from the hardware builtins, reusing the CPU target's
/// `#[cube]` helpers.
fn derive_positions(scope: &Scope, builtins: &mut BuiltinValues, cube_dim: Dim3) {
    let unit_pos_x = builtins.expect(Builtin::UnitPosX);
    let unit_pos_y = builtins.expect(Builtin::UnitPosY);
    let unit_pos_z = builtins.expect(Builtin::UnitPosZ);
    let cube_pos_x = builtins.expect(Builtin::CubePosX);
    let cube_pos_y = builtins.expect(Builtin::CubePosY);
    let cube_pos_z = builtins.expect(Builtin::CubePosZ);
    let cube_count_x = builtins.expect(Builtin::CubeCountX);
    let cube_count_y = builtins.expect(Builtin::CubeCountY);

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

/// The lane's index within its wavefront.
fn unit_pos_plane(scope: &Scope) -> Value {
    let (ops, lane) = lane_id_ops(scope.ctx_mut());
    for op in ops {
        scope.inserter().append_operation(scope.ctx(), op);
    }
    lane
}

/// Emits a call to LLVM intrinsic `name` returning `ret_ty`.
fn call_intrinsic(scope: &Scope, name: &str, ret_ty: TypeHandle) -> Value {
    let op = call_op(scope.ctx_mut(), name, ret_ty, vec![]);
    scope.register_with_result(&op)
}

fn call_i32_intrinsic(scope: &Scope, name: &str) -> Value {
    let ty = i32_ty(scope.ctx_mut());
    call_intrinsic(scope, name, ty)
}

/// `llvm.amdgcn.dispatch.ptr` returns a `ptr addrspace(4)` to the HSA kernel dispatch packet
/// the runtime prepared for this launch.
fn dispatch_ptr(scope: &Scope) -> Value {
    let ptr_ty = LlvmPointerType::get(scope.ctx_mut(), CONSTANT_ADDRESS_SPACE).into();
    call_intrinsic(scope, "llvm.amdgcn.dispatch.ptr", ptr_ty)
}

/// Loads a `u32` at `byte_offset` past `ptr`, via a single-index GEP over `i8` —
/// LLVM's only way to express "add N bytes".
fn load_u32_at(scope: &Scope, ptr: Value, byte_offset: u32) -> Value {
    let i8_ty = IntegerType::get(scope.ctx_mut(), 8, Signedness::Signless).into();
    let gep = GetElementPtrOp::new(
        scope.ctx_mut(),
        ptr,
        vec![GepIndex::Constant(byte_offset)],
        i8_ty,
    );
    let byte_ptr = scope.register_with_result(&gep);

    let u32_ty = i32_ty(scope.ctx_mut());
    let load = LoadOp::new(scope.ctx_mut(), byte_ptr, u32_ty);
    scope.register_with_result(&load)
}
