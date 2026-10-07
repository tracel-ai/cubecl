//! NVPTX builtins: the special registers the hardware hands each unit.

use crate::{
    prelude::*,
    shared::builtins::{LaunchValues, ReadsLaunchValues},
};

const TID: [&str; 3] = [
    "llvm.nvvm.read.ptx.sreg.tid.x",
    "llvm.nvvm.read.ptx.sreg.tid.y",
    "llvm.nvvm.read.ptx.sreg.tid.z",
];

const CTAID: [&str; 3] = [
    "llvm.nvvm.read.ptx.sreg.ctaid.x",
    "llvm.nvvm.read.ptx.sreg.ctaid.y",
    "llvm.nvvm.read.ptx.sreg.ctaid.z",
];

const NCTAID: [&str; 3] = [
    "llvm.nvvm.read.ptx.sreg.nctaid.x",
    "llvm.nvvm.read.ptx.sreg.nctaid.y",
    "llvm.nvvm.read.ptx.sreg.nctaid.z",
];

const LANEID: &str = "llvm.nvvm.read.ptx.sreg.laneid";

/// The PTX special registers `%tid`, `%ctaid`, `%nctaid` and `%laneid`.
#[derive(Debug)]
pub struct NvptxSpecialRegisters;

impl ReadsLaunchValues for NvptxSpecialRegisters {
    fn read(&self, scope: &Scope, _cube_dim: Dim3) -> LaunchValues {
        LaunchValues {
            unit_pos: TID.map(|register| read_sreg(scope, register)),
            cube_pos: CTAID.map(|register| read_sreg(scope, register)),
            cube_count: NCTAID.map(|register| read_sreg(scope, register)),
            unit_pos_plane: read_sreg(scope, LANEID),
        }
    }
}

fn read_sreg(scope: &Scope, name: &str) -> Value {
    let ty = i32_ty(scope.ctx_mut());
    let op = call_op(scope.ctx_mut(), name, ty, vec![]);
    scope.register_with_result(&op)
}
