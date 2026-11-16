//! Real kernels compiled for AMDGPU without a device, checked on the assembly.

use crate::shared::offline_kernels::{
    Wait, bf16_math_kernel, cdna_manual_product_kernel, cdna_quantized_product_kernel,
    fence_poll_kernel, fence_send_kernel, keep_largest_kernel, nested_calls_with_source_kernel,
    peer_tally_kernel, plane_moves_kernel, relay_kernel, scale_kernel, tally_kernel,
    tile_product_in_plane, tile_product_kernel,
};
use crate::target::LlvmTarget;
use crate::{
    AmdGpuModule, PlironArtifact, PlironCompiler, PlironOptions,
    amdgpu::codegen::{Assembly, compile_to_object},
};
use cubecl_core::Compiler;
use cubecl_core::ir::{AddressType, amd::GfxArch};
use cubecl_core::prelude::{MemoryOrder, SyncScope};
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
        let asm = asm_of(tile_product_kernel::<bf16, bf16, f32>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_f32_16x16x16_bf16"), "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<bf16, bf16, bf16>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_bf16_16x16x16_bf16"), "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<f16, f16, f16>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_f16_16x16x16_f16"), "{arch}:\n{asm}");
    }
}

/// RDNA4 multiplies `e4m3` (`fp8`) and `e5m2` (`bf8`) operands into `f32`, each side its own
/// format.
#[test]
fn fp8_tiles_multiply_on_rdna4_matrix_cores() {
    use cubecl_core::{e4m3, e5m2};
    let asm = asm_of(
        tile_product_kernel::<e4m3, e4m3, f32>((16, 16, 16)),
        "gfx1201",
    );
    assert!(asm.contains("v_wmma_f32_16x16x16_fp8_fp8"), "{asm}");
    let asm = asm_of(
        tile_product_kernel::<e4m3, e5m2, f32>((16, 16, 16)),
        "gfx1201",
    );
    assert!(asm.contains("v_wmma_f32_16x16x16_fp8_bf8"), "{asm}");
    let asm = asm_of(
        tile_product_kernel::<e5m2, e4m3, f32>((16, 16, 16)),
        "gfx1201",
    );
    assert!(asm.contains("v_wmma_f32_16x16x16_bf8_fp8"), "{asm}");
    let asm = asm_of(
        tile_product_kernel::<e5m2, e5m2, f32>((16, 16, 16)),
        "gfx1201",
    );
    assert!(asm.contains("v_wmma_f32_16x16x16_bf8_bf8"), "{asm}");
}

/// An 8-bit integer product tells the instruction which operand is signed through `neg_lo`, and
/// an unsigned pair sets none.
#[test]
fn int8_tiles_carry_each_operands_sign() {
    for arch in ["gfx1151", "gfx1201"] {
        let asm = asm_of(tile_product_kernel::<i8, u8, i32>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_i32_16x16x16_iu8"), "{arch}:\n{asm}");
        assert!(asm.contains("neg_lo:[1,0,0]"), "{arch}: a signed:\n{asm}");
        let asm = asm_of(tile_product_kernel::<u8, u8, i32>((16, 16, 16)), arch);
        assert!(asm.contains("v_wmma_i32_16x16x16_iu8"), "{arch}:\n{asm}");
        assert!(!asm.contains("neg_lo"), "{arch}: both unsigned:\n{asm}");
    }
}

/// A tile 32 deep is two 16-deep instructions: neither generation has a 32-deep form for these
/// operands.
#[test]
fn a_tile_two_instructions_deep_issues_two() {
    use cubecl_core::e4m3;
    use half::f16;
    let count = |asm: &str, op: &str| asm.matches(op).count();
    for arch in ["gfx1151", "gfx1201"] {
        let asm = asm_of(tile_product_kernel::<f16, f16, f32>((16, 16, 32)), arch);
        assert_eq!(count(&asm, "v_wmma_f32_16x16x16_f16"), 2, "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<i8, i8, i32>((16, 16, 32)), arch);
        assert_eq!(count(&asm, "v_wmma_i32_16x16x16_iu8"), 2, "{arch}:\n{asm}");
    }
    let asm = asm_of(
        tile_product_kernel::<e4m3, e4m3, f32>((16, 16, 32)),
        "gfx1201",
    );
    assert_eq!(count(&asm, "v_wmma_f32_16x16x16_fp8_fp8"), 2, "{asm}");
}

/// MI300 and MI200 multiply half-precision tiles on MFMA, one 16-deep instruction per 16 of `k`.
/// The assembler spells the same instructions differently on each.
#[test]
fn half_precision_tiles_multiply_on_mfma() {
    use half::{bf16, f16};
    let count = |asm: &str, op: &str| asm.matches(op).count();
    for (arch, f16_op, bf16_op) in [
        (
            "gfx942",
            "v_mfma_f32_16x16x16_f16",
            "v_mfma_f32_16x16x16_bf16",
        ),
        (
            "gfx90a",
            "v_mfma_f32_16x16x16f16",
            "v_mfma_f32_16x16x16bf16_1k",
        ),
    ] {
        let asm = asm_of(tile_product_kernel::<f16, f16, f32>((16, 16, 16)), arch);
        assert_eq!(count(&asm, f16_op), 1, "{arch}:\n{asm}");
        let asm = asm_of(tile_product_kernel::<bf16, bf16, f32>((16, 16, 32)), arch);
        assert_eq!(count(&asm, bf16_op), 2, "{arch}:\n{asm}");
    }
}

/// MI300's 8-bit integer MFMA is 32 deep, MI200's 16.
#[test]
fn int8_tiles_multiply_on_mfma() {
    let asm = asm_of(tile_product_kernel::<i8, i8, i32>((16, 16, 32)), "gfx942");
    assert_eq!(asm.matches("v_mfma_i32_16x16x32_i8").count(), 1, "{asm}");
    let asm = asm_of(tile_product_kernel::<i8, i8, i32>((16, 16, 32)), "gfx90a");
    assert_eq!(asm.matches("v_mfma_i32_16x16x16").count(), 2, "{asm}");
}

/// The architecture the MI300 tests compile for.
const MI300: &str = "gfx942";

/// The lanes of one MI300 plane.
const MI300_PLANE_DIM: u32 = 64;

/// Keeps `asm` as `<what>.s` in `CUBECL_OFFLINE_ASM_DIR`, where it is set, for reading.
fn keep_asm(asm: &str, what: &str) {
    let Some(dir) = std::env::var_os("CUBECL_OFFLINE_ASM_DIR") else {
        return;
    };
    let path = std::path::Path::new(&dir).join(format!("{what}.s"));
    std::fs::write(path, asm).unwrap();
}

/// What a kernel compiled for MI300 is checked for: `instructions` of `op`, and nothing on
/// scratch, which a fragment the lowering held in memory would read as.
struct Mfmas<'a> {
    op: &'a str,
    instructions: usize,
}

impl Mfmas<'_> {
    fn check(&self, asm: &str, what: &str) {
        keep_asm(asm, what);
        let count = asm.matches(self.op).count();
        assert_eq!(count, self.instructions, "{what}:\n{asm}");
        assert!(!asm.contains("scratch_"), "{what} spills:\n{asm}");
        assert!(
            asm.contains(".wavefront_size: 64"),
            "{what} is wave64:\n{asm}"
        );
    }
}

/// MI300's quantized MMA multiplies 8-bit float codes on its FNUZ MFMA, clearing negative zeros
/// first and scaling the sum into the accumulator with one `fma` a cell, and 8-bit integer codes
/// on its integer MFMA with no clearing, for a cube of one 64-lane plane and nothing on scratch.
#[test]
fn mi300_quantized_mma_compiles_to_mfma() {
    use cubecl_core::{
        e4m3, e5m2,
        ir::features::QuantizedMmaConfig,
        quant::scheme::{QuantValue, ScaleDtype},
        ue8m0,
    };
    let config = |a, b, k, block_k| QuantizedMmaConfig {
        a,
        b,
        scales: ScaleDtype::UE8M0,
        block_k,
        m: 16,
        n: 16,
        k,
    };
    // The clearing masks the low seven bits of every code, which nothing else does.
    let check = |asm: String, what: &str, mfmas: Mfmas, clears_negative_zeros: bool| {
        mfmas.check(&asm, what);
        let fma = asm.contains("v_fma_f32") || asm.contains("v_pk_fma_f32");
        assert!(fma, "{what} scales its sum with an fma:\n{asm}");
        let clears = asm.contains("0x7f7f7f7f");
        assert_eq!(clears, clears_negative_zeros, "{what} clears -0:\n{asm}");
    };

    let fp8_k32 = config(QuantValue::E4M3, QuantValue::E5M2, 32, 32);
    let asm = asm_of(
        cdna_quantized_product_kernel::<e4m3, e5m2, ue8m0>(fp8_k32),
        MI300,
    );
    let mfmas = Mfmas {
        op: "v_mfma_f32_16x16x32_fp8_bf8",
        instructions: 1,
    };
    check(asm, "quantized_fp8_k32", mfmas, true);

    let fp8_k64 = config(QuantValue::E4M3, QuantValue::E4M3, 64, 128);
    let asm = asm_of(
        cdna_quantized_product_kernel::<e4m3, e4m3, ue8m0>(fp8_k64),
        MI300,
    );
    let mfmas = Mfmas {
        op: "v_mfma_f32_16x16x32_fp8_fp8",
        instructions: 2,
    };
    check(asm, "quantized_fp8_k64", mfmas, true);

    let int8_k32 = config(QuantValue::Q8S, QuantValue::Q8S, 32, 64);
    let asm = asm_of(
        cdna_quantized_product_kernel::<i8, i8, ue8m0>(int8_k32),
        MI300,
    );
    let mfmas = Mfmas {
        op: "v_mfma_i32_16x16x32_i8",
        instructions: 1,
    };
    check(asm, "quantized_i8_k32", mfmas, false);
}

/// Every form MI300 offers compiles for a cube of one 64-lane plane, through the fragment API and
/// the manual one alike, to one MFMA per instruction deep and nothing on scratch.
#[test]
fn every_mi300_form_compiles_for_a_wave64_plane() {
    use half::{bf16, f16};
    for (k, instructions) in [(16, 1), (32, 2)] {
        let shape = (16, 16, k);
        let f16_mfmas = Mfmas {
            op: "v_mfma_f32_16x16x16_f16",
            instructions,
        };
        let bf16_mfmas = Mfmas {
            op: "v_mfma_f32_16x16x16_bf16",
            instructions,
        };
        let asm = asm_of(
            tile_product_in_plane::<f16, f16, f32>(shape, MI300_PLANE_DIM),
            MI300,
        );
        f16_mfmas.check(&asm, &format!("cmma_f16_k{k}"));
        let asm = asm_of(
            tile_product_in_plane::<bf16, bf16, f32>(shape, MI300_PLANE_DIM),
            MI300,
        );
        bf16_mfmas.check(&asm, &format!("cmma_bf16_k{k}"));
        let asm = asm_of(cdna_manual_product_kernel::<f16, f16, f32>(shape), MI300);
        f16_mfmas.check(&asm, &format!("manual_f16_k{k}"));
        let asm = asm_of(cdna_manual_product_kernel::<bf16, bf16, f32>(shape), MI300);
        bf16_mfmas.check(&asm, &format!("manual_bf16_k{k}"));
    }
    for (k, instructions) in [(32, 1), (64, 2)] {
        let shape = (16, 16, k);
        let mfmas = Mfmas {
            op: "v_mfma_i32_16x16x32_i8",
            instructions,
        };
        let asm = asm_of(
            tile_product_in_plane::<i8, i8, i32>(shape, MI300_PLANE_DIM),
            MI300,
        );
        mfmas.check(&asm, &format!("cmma_i8_k{k}"));
        let asm = asm_of(cdna_manual_product_kernel::<i8, i8, i32>(shape), MI300);
        mfmas.check(&asm, &format!("manual_i8_k{k}"));
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

/// A `Peer` buffer may be written by another device while the kernel runs, so it is promised
/// nothing, and an atomic on it synchronizes at system scope and is not promised device-local
/// memory, which would make a float add one that is not atomic across the link. The kernel's
/// own buffers keep their promises.
#[test]
fn a_peer_buffer_is_promised_nothing() {
    let params = entry_params(peer_tally_kernel(), "gfx1151");
    assert!(
        !params[2].contains("noalias") && !params[2].contains("readonly"),
        "a peer may write the tally:\n{params:#?}"
    );
    assert!(
        params[0].contains("noalias") && params[0].contains("readonly"),
        "the input:\n{params:#?}"
    );
    assert!(params[1].contains("noalias"), "the output:\n{params:#?}");
    let module = module_of(peer_tally_kernel(), &GfxArch::parse("gfx1151"));
    let add = module
        .ir
        .lines()
        .find(|line| line.contains("atomicrmw add"))
        .expect("the kernel counts into the tally");
    assert!(
        !add.contains("syncscope"),
        "an atomic on a peer buffer synchronizes at system scope: {add}"
    );
    assert!(
        !module.ir.contains("amdgpu.no.remote.memory"),
        "an atomic of a peer kernel may touch remote memory:\n{}",
        module.ir
    );
}

/// A release fence before a flag is a write-back of the caches and an acquire after it an
/// invalidation, both at system scope and with no barrier.
#[test]
fn a_unit_fence_is_a_cache_operation_without_a_barrier() {
    let line_of = |asm: &str, op: &str| {
        asm.lines()
            .position(|line| line.contains(op) && line.contains("scope:SCOPE_SYS"))
            .unwrap_or_else(|| panic!("{op} at system scope:\n{asm}"))
    };
    let send = asm_of(
        fence_send_kernel(MemoryOrder::Release, SyncScope::System),
        "gfx1201",
    );
    let write_back = line_of(&send, "global_wb");
    let flag = line_of(&send, "global_store_b32");
    assert!(write_back < flag, "the flag follows the release:\n{send}");

    let poll = asm_of(
        fence_poll_kernel(MemoryOrder::Acquire, SyncScope::System),
        "gfx1201",
    );
    let load = line_of(&poll, "global_load_b32");
    let invalidate = line_of(&poll, "global_inv");
    assert!(load < invalidate, "the acquire follows the load:\n{poll}");
}

/// Each scope is LLVM's syncscope of that name, unnamed for the system, a unit's own scope emits
/// no fence, and a fence is never a barrier.
#[test]
fn a_unit_fence_lowers_to_the_syncscope_of_its_scope() {
    let table = [
        (SyncScope::Unit, None),
        (SyncScope::Plane, Some("syncscope(\"wavefront\") ")),
        (SyncScope::Cube, Some("syncscope(\"workgroup\") ")),
        (SyncScope::Device, Some("syncscope(\"agent\") ")),
        (SyncScope::System, Some("")),
    ];
    fn check(kernel: impl CubeKernel, sync: SyncScope, name: Option<&str>, order: &str) {
        let module = module_of(kernel, &GfxArch::parse("gfx1201"));
        let fenced = module
            .ir
            .lines()
            .any(|line| line.trim_start().starts_with("fence "));
        match name {
            Some(name) => assert!(
                module.ir.contains(&format!("fence {name}{order}")),
                "{sync:?} {order}:\n{}",
                module.ir
            ),
            None => assert!(!fenced, "{sync:?} {order}:\n{}", module.ir),
        }
        assert!(!module.ir.contains("s.barrier"), "{}", module.ir);
    }
    for (sync, name) in table {
        check(
            fence_send_kernel(MemoryOrder::Release, sync),
            sync,
            name,
            "release",
        );
        check(
            fence_poll_kernel(MemoryOrder::Acquire, sync),
            sync,
            name,
            "acquire",
        );
        check(
            fence_send_kernel(MemoryOrder::AcqRel, sync),
            sync,
            name,
            "acq_rel",
        );
    }
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

/// An `e2m1x2` cast compiles on a part with no fp4 conversion of its own, a Strix Halo's: the
/// software codec, no fp4 conversion instruction.
#[test]
fn fp4_casts_compile_to_the_software_codec() {
    let asm = asm_of(
        crate::shared::offline_kernels::fp4_round_trip_kernel(),
        "gfx1151",
    );
    let fp4_conversion = asm
        .lines()
        .any(|line| line.trim_start().starts_with("v_cvt") && line.contains("fp4"));
    assert!(!fp4_conversion, "no fp4 conversion instruction:\n{asm}");
    assert!(
        asm.contains("v_lshr") || asm.contains("v_bfe"),
        "the codes are shifted out:\n{asm}"
    );
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
