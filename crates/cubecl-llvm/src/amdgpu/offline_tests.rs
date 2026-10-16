//! Real kernels compiled for AMDGPU without a device, checked on the assembly.

use crate::shared::offline_kernels::{
    Wait, bf16_math_kernel, keep_largest_kernel, nested_calls_with_source_kernel,
    plane_moves_kernel, relay_kernel, scale_kernel, tally_kernel, tile_product_kernel,
};
use crate::target::LlvmTarget;
use crate::{
    AmdGpuModule, PlironArtifact, PlironCompiler, PlironOptions,
    amdgpu::codegen::{Assembly, compile_to_object},
};
use cubecl_core::Compiler;
use cubecl_core::ir::{AddressType, amd::GfxArch};
use cubecl_core::runtime_tests::offline::SOURCE_PATH;
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

/// Full debug data embeds the kernel source, with its MD5, in the DWARF 5 line table.
#[test]
fn full_debug_info_embeds_the_source_text() {
    let asm = asm_of(nested_calls_with_source_kernel(), "gfx1201");
    let file = asm
        .lines()
        .find(|line| line.contains(SOURCE_PATH) && line.contains(" source "))
        .unwrap_or_else(|| panic!("no `.file` with a source text:\n{asm}"));
    assert!(file.contains(" md5 0x"), "{file}");
    assert!(file.contains("fn nested_calls_with_source"), "{file}");
}

/// CDNA2 and RDNA have no `bf16` arithmetic, so LLVM computes a `bf16` kernel in `f32` there.
#[test]
fn bf16_computes_in_f32_without_a_bf16_alu() {
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

/// A cube that waits on a counter reads what the cube before it wrote within the launch: the
/// buffers some cube writes are promised nothing, or a read after the handoff could be a scalar
/// load from a cache the acquire leaves stale. The weights no cube writes keep both promises.
#[test]
fn a_relay_promises_nothing_about_what_its_cubes_write() {
    for wait in [Wait::Load, Wait::CompareExchange] {
        let params = entry_params(relay_kernel(wait), "gfx1151");
        for (binding, name) in [(0, "turns"), (1, "carry"), (3, "out")] {
            assert!(
                !params[binding].contains("noalias"),
                "{wait:?}: some cube writes `{name}` within the launch:\n{params:#?}"
            );
        }
        assert!(
            params[2].contains("noalias") && params[2].contains("readonly"),
            "{wait:?}: no cube writes the weights:\n{params:#?}"
        );
    }
}

/// An `atomic_add` whose result is dropped observes no other cube, so the buffers beside it keep
/// their promises.
#[test]
fn a_dropped_atomic_add_keeps_the_promises() {
    let params = entry_params(tally_kernel(), "gfx1151");
    assert!(
        params[0].contains("noalias") && params[0].contains("readonly"),
        "the input:\n{params:#?}"
    );
    assert!(params[1].contains("noalias"), "the output:\n{params:#?}");
}

#[test]
fn a_kernel_without_atomics_keeps_its_promises() {
    let params = entry_params(scale_kernel(AddressType::U32), "gfx1151");
    assert!(
        params[0].contains("noalias") && params[0].contains("readonly"),
        "the input:\n{params:#?}"
    );
    assert!(params[1].contains("noalias"), "the output:\n{params:#?}");
}

/// The assembly `kernel` compiles to for `arch`.
fn asm_of(kernel: impl CubeKernel, arch: &str) -> String {
    let arch = GfxArch::parse(arch);
    let module = module_of(kernel, &arch);
    compile_to_object(&module.ir, &arch, Assembly::Keep)
        .unwrap()
        .1
        .unwrap()
}

/// The attributes on each parameter of the entry point `kernel` finalizes to for `arch`.
fn entry_params(kernel: impl CubeKernel, arch: &str) -> Vec<String> {
    let module = module_of(kernel, &GfxArch::parse(arch));
    let signature = module
        .ir
        .lines()
        .find(|line| line.starts_with("define"))
        .expect("the module defines its entry point");
    let open = signature.find('(').unwrap();
    let close = signature.rfind(')').unwrap();
    signature[open + 1..close]
        .split(',')
        .map(str::to_string)
        .collect()
}

/// The module the pliron compiler produces for `kernel` on `arch`.
fn module_of(kernel: impl CubeKernel, arch: &GfxArch) -> AmdGpuModule {
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
    module
}
