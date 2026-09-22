//! Real kernels compiled to PTX without a device, checked on the assembly.

use crate::shared::offline_kernels::{plane_moves_kernel, scale_kernel};
use crate::target::LlvmTarget;
use crate::{PlironArtifact, PlironCompiler, PlironOptions, nvptx::ptx_version::PtxVersion};
use cubecl_core::Compiler;
use cubecl_core::ir::{AddressType, nvidia::SmArch};
use cubecl_runtime::kernel::CubeKernel;
use std::ffi::CStr;

/// The PTX `kernel` compiles to for `sm_{arch}`.
fn ptx_of(kernel: impl CubeKernel, arch: u32) -> String {
    let mut compiler = PlironCompiler {
        target: LlvmTarget::Nvptx,
    };
    let options = PlironOptions {
        sm_arch: Some(SmArch::new(arch, false)),
        ptx_version: PtxVersion::for_driver(12080),
        ..Default::default()
    };
    let PlironArtifact::NvptxCode(module) = compiler.compile(kernel.define(), &options).unwrap()
    else {
        unreachable!("the NVPTX target produces PTX");
    };
    // SAFETY: the module's PTX is NUL-terminated.
    unsafe { CStr::from_ptr(module.ptx.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_1d_cube_reads_only_the_x_thread_id() {
    let ptx = ptx_of(scale_kernel(AddressType::U32), 60);
    assert!(
        ptx.contains(".reqntid 64, 1, 1"),
        "exact launch bounds:\n{ptx}"
    );
    assert!(ptx.contains("%tid.x"), "{ptx}");
    assert!(
        !ptx.contains("%tid.y") && !ptx.contains("%tid.z"),
        "an axis of one unit is zero, not a register:\n{ptx}"
    );
}

#[test]
fn plane_moves_are_native_shuffles() {
    let ptx = ptx_of(plane_moves_kernel(), 60);
    assert!(ptx.contains("shfl.sync.idx"), "the broadcast:\n{ptx}");
    assert!(ptx.contains("shfl.sync.bfly"), "the XOR:\n{ptx}");
}
