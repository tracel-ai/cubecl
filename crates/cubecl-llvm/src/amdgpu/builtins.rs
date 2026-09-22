//! AMDGPU builtins: the work-item and workgroup ids, and the grid size in the dispatch packet.

use crate::{
    amdgpu::intrinsic::lane_id_ops,
    prelude::*,
    shared::builtins::{LaunchIds, LaunchRegisters},
};
use cubecl_core::prelude::*;
use pliron_llvm::ops::{GepIndex, GetElementPtrOp, LoadOp};

const WORKITEM_ID: [&str; 3] = [
    "llvm.amdgcn.workitem.id.x",
    "llvm.amdgcn.workitem.id.y",
    "llvm.amdgcn.workitem.id.z",
];

const WORKGROUP_ID: [&str; 3] = [
    "llvm.amdgcn.workgroup.id.x",
    "llvm.amdgcn.workgroup.id.y",
    "llvm.amdgcn.workgroup.id.z",
];

/// The HSA dispatch packet uses the constant address space.
const CONSTANT_ADDRESS_SPACE: u32 = 4;

/// Byte offsets of the work-item grid dimensions in the HSA dispatch packet.
const GRID_SIZE_OFFSETS: [u32; 3] = [12, 16, 20];

/// The work-item and workgroup id intrinsics, the lane from `mbcnt`, and the cube count from
/// the dispatch packet, which gives the grid in work-items rather than workgroups.
#[derive(Debug)]
pub struct AmdGpuRegisters;

impl LaunchRegisters for AmdGpuRegisters {
    fn read(&self, scope: &Scope, cube_dim: Dim3) -> LaunchIds {
        let unit_pos = WORKITEM_ID.map(|intrinsic| call_i32_intrinsic(scope, intrinsic));
        let cube_pos = WORKGROUP_ID.map(|intrinsic| call_i32_intrinsic(scope, intrinsic));
        let unit_pos_plane = unit_pos_plane(scope);

        let dispatch_ptr = dispatch_ptr(scope);
        let [x, y, z] = GRID_SIZE_OFFSETS.map(|offset| load_u32_at(scope, dispatch_ptr, offset));
        let cube_count =
            [(x, cube_dim.x), (y, cube_dim.y), (z, cube_dim.z)].map(|(grid_size, dim)| {
                cube_count_component::expand(scope, grid_size.into(), dim).value(scope)
            });

        LaunchIds {
            unit_pos,
            cube_pos,
            cube_count,
            unit_pos_plane,
        }
    }
}

#[cube]
fn cube_count_component(grid_size: u32, #[comptime] cube_dim: u32) -> u32 {
    grid_size / cube_dim
}

fn unit_pos_plane(scope: &Scope) -> Value {
    let (ops, lane) = lane_id_ops(scope.ctx_mut());
    for op in ops {
        scope.inserter().append_operation(scope.ctx(), op);
    }
    lane
}

fn call_intrinsic(scope: &Scope, name: &str, ret_ty: TypeHandle) -> Value {
    let op = call_op(scope.ctx_mut(), name, ret_ty, vec![]);
    scope.register_with_result(&op)
}

fn call_i32_intrinsic(scope: &Scope, name: &str) -> Value {
    let ty = i32_ty(scope.ctx_mut());
    call_intrinsic(scope, name, ty)
}

fn dispatch_ptr(scope: &Scope) -> Value {
    let ptr_ty = LlvmPointerType::get(scope.ctx_mut(), CONSTANT_ADDRESS_SPACE).into();
    call_intrinsic(scope, "llvm.amdgcn.dispatch.ptr", ptr_ty)
}

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
