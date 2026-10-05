//! Real kernels compiled for AMDGPU without a device, checked on the assembly.

use crate::shared::offline_kernels::{
    bf16_math_kernel, keep_largest_kernel, plane_moves_kernel, scale_kernel, tile_product_kernel,
};
use crate::target::LlvmTarget;
use crate::{
    PlironArtifact, PlironCompiler, PlironOptions,
    amdgpu::codegen::{Assembly, compile_to_object},
};
use cubecl_core::Compiler;
use cubecl_core::ir::{AddressType, amd::GfxArch};
use cubecl_runtime::kernel::CubeKernel;

#[test]
fn a_32_bit_kernel_indexes_in_32_bits() {
    let asm = asm_of(scale_kernel(AddressType::U32), "gfx1201");
    assert!(
        !asm.contains("s_mul_u64") && !asm.contains("_u64_e32"),
        "no 64-bit index arithmetic:\n{asm}"
    );
    assert!(
        asm.contains("v_cmpx_gt_u32"),
        "a 32-bit bounds check:\n{asm}"
    );
}

#[test]
fn a_64_bit_kernel_indexes_in_64_bits() {
    let asm = asm_of(scale_kernel(AddressType::U64), "gfx1201");
    assert!(asm.contains("_u64"), "a 64-bit bounds check:\n{asm}");
}

/// Every shuffle is lowered to `ds_bpermute`, and the backend is what turns one with a known
/// source lane into a register move: `v_readlane` for a broadcast, DPP for a small XOR. A
/// change that hid the lane from it would put every such move back through LDS.
#[test]
fn a_constant_lane_moves_without_lds() {
    let asm = asm_of(plane_moves_kernel(), "gfx1201");
    assert!(
        asm.contains("v_readlane_b32"),
        "the broadcast is a readlane:\n{asm}"
    );
    assert!(
        asm.contains("quad_perm:[1,0,3,2]"),
        "the XOR is DPP:\n{asm}"
    );
    assert!(
        !asm.contains("ds_bpermute"),
        "nothing goes through LDS:\n{asm}"
    );
}

#[test]
fn a_1d_cube_ignores_the_y_and_z_work_item_ids() {
    let asm = asm_of(scale_kernel(AddressType::U32), "gfx1201");
    // The three ids arrive packed in `v0`, 10 bits each: y at bit 10, z at bit 20.
    assert!(!asm.contains("v_bfe_u32"), "no id is unpacked:\n{asm}");
}

/// A loop of a constant trip count that indexes a local array is unrolled, so every index is a
/// constant and the array becomes VGPRs rather than scratch memory.
#[test]
fn a_local_array_under_a_constant_loop_is_registers() {
    let asm = asm_of(keep_largest_kernel(64), "gfx1201");
    assert!(
        !asm.contains("scratch_"),
        "the array is in scratch memory:\n{asm}"
    );
}

/// `bf16` lanes move as 16-bit integers and compute in `f32`, on CDNA and RDNA alike.
#[test]
fn bf16_moves_as_bits_and_computes_in_f32() {
    for arch in ["gfx90a", "gfx1151"] {
        let asm = asm_of(bf16_math_kernel(), arch);
        assert!(asm.contains("v_sqrt_f32"), "the math runs in f32:\n{asm}");
    }
}

/// A half-precision tile product is one WMMA instruction on RDNA3 and RDNA4, into an `f32` or
/// a 16-bit accumulator.
#[test]
fn half_precision_tiles_multiply_on_the_matrix_cores() {
    use half::{bf16, f16};
    for arch in ["gfx1151", "gfx1201"] {
        let asm = asm_of(tile_product_kernel::<bf16, f32>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_f32_16x16x16_bf16"), "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<bf16, bf16>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_bf16_16x16x16_bf16"), "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<f16, f16>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_f16_16x16x16_f16"), "{arch}:\n{asm}");
    }
}

/// The assembly `kernel` compiles to for `arch`.
fn asm_of(kernel: impl CubeKernel, arch: &str) -> String {
    let arch = GfxArch::parse(arch);
    let mut compiler = PlironCompiler {
        target: LlvmTarget::AmdGpu,
    };
    let options = PlironOptions {
        arch: Some(arch.clone()),
        ..Default::default()
    };
    let PlironArtifact::AmdGpuCode(module) = compiler.compile(kernel.define(), &options).unwrap()
    else {
        unreachable!("the AMDGPU target produces a code object");
    };
    compile_to_object(&module.ir, &arch, Assembly::Keep)
        .unwrap()
        .1
        .unwrap()
}
