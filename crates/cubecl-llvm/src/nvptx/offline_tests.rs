//! Real kernels compiled to PTX without a device, checked on the assembly.

use crate::shared::offline_kernels::{
    keep_largest_kernel, plane_moves_kernel, scale_kernel, strided_walk_kernel,
};
use crate::target::LlvmTarget;
use crate::{PlironArtifact, PlironCompiler, PlironOptions, nvptx::ptx_version::PtxVersion};
use cubecl_core::Compiler;
use cubecl_core::ir::{AddressType, nvidia::SmArch};
use cubecl_runtime::kernel::CubeKernel;
use std::ffi::CStr;

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
    // Volta and later schedule a plane's lanes independently, so a full member mask is
    // undefined inside a branch only some of them take.
    assert!(
        ptx.contains("activemask.b32"),
        "the executing lanes as the mask:\n{ptx}"
    );
}

/// A loop of a constant trip count that indexes a local array is unrolled, so every index is a
/// constant and the array becomes registers rather than local memory.
#[test]
fn a_local_array_under_a_constant_loop_is_registers() {
    let ptx = ptx_of(keep_largest_kernel(64), 60);
    assert!(
        !ptx.contains("ld.local") && !ptx.contains("st.local"),
        "the array is in local memory:\n{ptx}"
    );
}

/// The address of a strided walk advances by an add each iteration. Rebuilding it from the loop
/// counter instead costs a multiply per load, and `ptxas` then interleaves the loads with the
/// arithmetic that waits on them.
#[test]
fn a_strided_walk_advances_its_address() {
    let ptx = ptx_of(strided_walk_kernel(), 75);
    let body = loop_body(&ptx).expect("the walk is a loop");
    assert!(
        !body.contains("mad.lo") && !body.contains("mul.lo"),
        "the address is rebuilt from the counter:\n{body}"
    );
}

/// The instructions from the first label to the branch that jumps back to it.
fn loop_body(ptx: &str) -> Option<String> {
    let lines: Vec<&str> = ptx.lines().collect();
    for (start, line) in lines.iter().enumerate() {
        let Some(label) = line.strip_suffix(':') else {
            continue;
        };
        let back_edge = format!("bra \t{label};");
        if let Some(end) = lines[start..].iter().position(|l| l.contains(&back_edge)) {
            return Some(lines[start..=start + end].join("\n"));
        }
    }
    None
}

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
fn tf32_vector_casts_round_each_lane() {
    let ptx = ptx_of(crate::shared::offline_kernels::tf32_round_kernel(), 80);
    assert_eq!(
        ptx.matches("cvt.rna.tf32.f32").count(),
        4,
        "TF32 casts must round every lane:\n{ptx}"
    );
}

#[test]
fn tf32_constant_casts_preserve_rounding() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::tf32_round_constants_kernel(),
        80,
    );
    assert_eq!(
        ptx.matches("cvt.rna.tf32.f32").count(),
        4,
        "constant float and integer casts must retain TF32 rounding through an FP32 round trip:\n{ptx}"
    );
}
