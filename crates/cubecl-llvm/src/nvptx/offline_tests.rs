//! Real kernels compiled to PTX without a device, checked on the assembly.

use crate::shared::offline_kernels::{
    Wait, bf16_math_kernel, keep_largest_kernel, plane_moves_kernel, relay_kernel, scale_kernel,
    strided_walk_kernel, tally_kernel, tile_product_kernel,
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

/// `bf16` arithmetic runs on Ampere's packed `bf16x2` instructions and converts with one `cvt`;
/// a part without them computes in `f32`.
#[test]
fn bf16_is_native_where_the_part_has_it() {
    let ampere = ptx_of(bf16_math_kernel(), 80);
    assert!(
        ampere.contains("fma.rn.bf16x2"),
        "packed bf16 arithmetic:\n{ampere}"
    );
    assert!(
        ampere.contains("cvt.rn.bf16.f32"),
        "a native conversion:\n{ampere}"
    );

    let pascal = ptx_of(bf16_math_kernel(), 60);
    assert!(!pascal.contains("bf16x2"), "{pascal}");
    assert!(pascal.contains("fma.rn.f32"), "promoted to f32:\n{pascal}");
}

/// A `bf16` tile product is one tensor core instruction on Ampere, accumulating in `f32`, for
/// every tile WMMA has: the tiles that are not square size A and B apart.
#[test]
fn bf16_tiles_multiply_on_the_tensor_cores() {
    for (m, n, k) in [(16, 16, 16), (32, 8, 16), (8, 32, 16)] {
        let ptx = ptx_of(tile_product_kernel::<half::bf16, f32>((m, n, k)), 80);
        let mma = format!("wmma.mma.sync.aligned.row.col.m{m}n{n}k{k}.f32.bf16.bf16.f32");
        assert!(ptx.contains(&mma), "{ptx}");
    }
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

/// A cube that waits on a counter reads what the cube before it wrote within the launch: the
/// carry some cube writes is loaded coherently, and the weights no cube writes through the
/// non-coherent cache.
#[test]
fn a_relay_loads_what_its_cubes_write_coherently() {
    for wait in [Wait::Load, Wait::CompareExchange] {
        let ptx = ptx_of(relay_kernel(wait), 70);
        let non_coherent = ptx.matches("ld.global.nc").count();
        assert_eq!(
            non_coherent, 1,
            "{wait:?}: the weights alone go through the non-coherent cache:\n{ptx}"
        );
    }
}

/// An `atomic_add` whose result is dropped observes no other cube, so the input beside it still
/// loads through the non-coherent cache.
#[test]
fn a_dropped_atomic_add_keeps_the_non_coherent_load() {
    let ptx = ptx_of(tally_kernel(), 70);
    assert!(ptx.contains("ld.global.nc"), "the input:\n{ptx}");
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

/// The module `kernel` compiles to for `sm_{arch}`.
fn module_of(kernel: impl CubeKernel, arch: u32) -> crate::NvptxModule {
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
    module
}

/// The PTX `kernel` compiles to for `sm_{arch}`.
fn ptx_of(kernel: impl CubeKernel, arch: u32) -> String {
    let module = module_of(kernel, arch);
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

/// A warpgroup MMA is one `wgmma.mma_async` between the fence and the group it is committed in,
/// reading the tiles the units staged after fencing the async proxy.
#[test]
fn a_warpgroup_product_is_one_async_mma() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::warpgroup_product_kernel::<half::f16, f32>(64, 16, false),
        90,
    );
    assert!(ptx.contains(".target sm_90a"), "{ptx}");
    assert!(ptx.contains("fence.proxy.async.shared::cta"), "{ptx}");
    assert!(ptx.contains("wgmma.fence.sync.aligned"), "{ptx}");
    assert_eq!(
        ptx.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16")
            .count(),
        1,
        "{ptx}"
    );
    assert!(ptx.contains("wgmma.commit_group.sync.aligned"), "{ptx}");
    assert!(ptx.contains("wgmma.wait_group.sync.aligned \t0;"), "{ptx}");
    assert!(
        !ptx.contains("ld.local") && !ptx.contains("st.local"),
        "the accumulator is registers:\n{ptx}"
    );
}

/// `A` in registers is four 32-bit registers per unit, and the 16-bit forms name only `B`'s
/// transpose.
#[test]
fn a_warpgroup_product_takes_a_from_registers() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::warpgroup_product_kernel::<half::bf16, f32>(32, 16, true),
        90,
    );
    let mma = ptx
        .lines()
        .find(|line| line.contains("wgmma.mma_async"))
        .unwrap_or_else(|| panic!("no MMA:\n{ptx}"));
    assert!(mma.contains("m64n32k16.f32.bf16.bf16"), "{mma}");
    // The 16 accumulator registers, then the four of `A`.
    assert_eq!(mma.matches('{').count(), 2, "{mma}");
}

/// The integer forms take neither the operand negation nor the transposes.
#[test]
fn an_integer_warpgroup_product_has_no_immediates() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::warpgroup_product_kernel::<i8, i32>(32, 32, false),
        90,
    );
    let mma = ptx
        .lines()
        .find(|line| line.contains("wgmma.mma_async"))
        .unwrap_or_else(|| panic!("no MMA:\n{ptx}"));
    assert!(mma.contains("m64n32k32.s32.s8.s8"), "{mma}");
    assert!(mma.trim_end().ends_with(", p;"), "{mma}");
}

/// PTX wants every write to `A`'s registers ahead of the `wgmma.fence` before the MMA that reads
/// them. Pinned by `fence_operand`, the compiler cannot sink them past it.
#[test]
fn the_registers_a_warpgroup_product_reads_are_written_before_its_fence() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::warpgroup_product_kernel::<half::f16, f32>(32, 16, true),
        90,
    );
    let mma = ptx
        .find("wgmma.mma_async")
        .unwrap_or_else(|| panic!("no MMA:\n{ptx}"));
    let fence = ptx[..mma]
        .rfind("wgmma.fence.sync.aligned")
        .unwrap_or_else(|| panic!("no fence before the MMA:\n{ptx}"));
    let between = &ptx[fence..mma];
    let mma_line = ptx[mma..].lines().next().expect("the MMA's line");
    // The operands are the accumulator's registers, then `A`'s.
    let a_registers = mma_line
        .split('{')
        .nth(2)
        .and_then(|operands| operands.split('}').next())
        .unwrap_or_else(|| panic!("no A registers: {mma_line}"));
    for register in a_registers.split(',').map(str::trim) {
        let written = between
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(&format!("{register},")));
        assert!(
            !written,
            "{register} written between the fence and the MMA:\n{between}"
        );
    }
}

/// The `n` of each wait on a commit group of `kind`, in program order.
fn group_waits(ptx: &str, kind: &str) -> Vec<u32> {
    ptx.lines()
        .filter_map(|line| line.trim().strip_prefix(kind))
        .map(|n| {
            n.trim()
                .trim_end_matches(';')
                .parse()
                .unwrap_or_else(|_| panic!("a count: {n}"))
        })
        .collect()
}

/// Each wait lets run the fewest groups committed after its token's on any path to it.
#[test]
fn a_wait_counts_the_groups_on_the_shortest_path() {
    use crate::shared::offline_kernels::{WaitCase, group_waits_kernel};

    // LLVM may unroll or duplicate a wait, so every copy of it must agree.
    let bulk = |case| {
        let ptx = ptx_of(group_waits_kernel(case), 90);
        let waits = group_waits(&ptx, "cp.async.bulk.wait_group.read");
        assert!(!waits.is_empty(), "no wait in {case:?}:\n{ptx}");
        assert!(waits.iter().all(|&n| n == waits[0]), "{case:?}: {waits:?}");
        waits[0]
    };
    // The path around the branch commits nothing.
    assert_eq!(bulk(WaitCase::OneBranchStores), 0);
    assert_eq!(bulk(WaitCase::BothBranchesStore), 1);
    // Both stores of the iteration follow the previous iteration's second, or the store before
    // the loop.
    assert_eq!(bulk(WaitCase::TwoStoresPerIteration), 2);
    assert_eq!(bulk(WaitCase::MaybeEmptyInnerLoop), 0);
    // A warpgroup group is no bulk group.
    assert_eq!(bulk(WaitCase::OtherKind), 0);
    let ptx = ptx_of(group_waits_kernel(WaitCase::OtherKind), 90);
    assert_eq!(group_waits(&ptx, "wgmma.wait_group.sync.aligned"), [0]);
}

/// A tensor map is a 128-byte parameter the kernel reads in place: had LLVM copied it to the
/// stack, the copy would be in local memory, which TMA cannot read.
#[test]
fn a_tensor_map_is_a_grid_constant_parameter() {
    let ptx = ptx_of(crate::shared::offline_kernels::tma_tile_load_kernel(), 90);
    assert!(
        ptx.contains(".param .align 64 .b8 tma_tile_load"),
        "the map is passed by value:\n{ptx}"
    );
    assert!(
        !ptx.contains("st.local") && !ptx.contains("ld.local"),
        "the map was copied to local memory:\n{ptx}"
    );

    // The map is no buffer, so it gets none of their aliasing promises, while the output does.
    let ir = module_of(crate::shared::offline_kernels::tma_tile_load_kernel(), 90).ir;
    let define = ir
        .lines()
        .find(|line| line.starts_with("define"))
        .unwrap_or_else(|| panic!("no entry point:\n{ir}"));
    let (map, rest) = define
        .split_once("%0,")
        .unwrap_or_else(|| panic!("no map parameter: {define}"));
    assert!(map.contains("nvvm.grid_constant"), "{map}");
    assert!(
        !map.contains("noalias") && !map.contains("readonly"),
        "{map}"
    );
    assert!(rest.contains("noalias"), "{rest}");
}

/// A tiled load completes on the barrier as a transaction: the units expect its bytes, arrive,
/// and spin until the phase completes.
#[test]
fn a_tma_load_completes_on_an_mbarrier() {
    let ptx = ptx_of(crate::shared::offline_kernels::tma_tile_load_kernel(), 90);
    assert!(ptx.contains("mbarrier.init.shared"), "{ptx}");
    assert!(
        ptx.contains(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ),
        "{ptx}"
    );
    assert!(ptx.contains("mbarrier.expect_tx"), "{ptx}");
    assert!(ptx.contains("mbarrier.arrive"), "{ptx}");
    assert!(ptx.contains("mbarrier.try_wait.shared::cta.b64"), "{ptx}");
}

/// Each store is its own bulk group. The first store's wait lets the second's group run on,
/// and the second's waits for both.
#[test]
fn a_tma_store_is_tracked_by_a_bulk_group() {
    let ptx = ptx_of(crate::shared::offline_kernels::tma_tile_store_kernel(), 90);
    assert!(ptx.contains("fence.proxy.async.shared::cta"), "{ptx}");
    assert_eq!(
        ptx.matches("cp.async.bulk.tensor.2d.global.shared::cta")
            .count(),
        2,
        "{ptx}"
    );
    assert_eq!(
        ptx.matches("cp.async.bulk.commit_group").count(),
        2,
        "{ptx}"
    );
    let waits: Vec<&str> = ptx
        .lines()
        .filter(|line| line.contains("cp.async.bulk.wait_group.read"))
        .map(str::trim)
        .collect();
    assert_eq!(
        waits,
        [
            "cp.async.bulk.wait_group.read \t1;",
            "cp.async.bulk.wait_group.read \t0;"
        ],
        "{ptx}"
    );
}

/// In a loop that commits a stage of MMAs per iteration, the wait on the previous stage lets
/// the stage just committed run on, and the accumulator's wait drains every group.
#[test]
fn a_pipelined_wait_lets_the_newest_stage_run() {
    let ptx = ptx_of(
        crate::shared::offline_kernels::warpgroup_pipeline_kernel(),
        90,
    );
    let body = loop_body(&ptx).unwrap_or_else(|| panic!("no loop:\n{ptx}"));
    assert_eq!(
        body.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16")
            .count(),
        4,
        "{body}"
    );
    assert!(body.contains("wgmma.commit_group.sync.aligned"), "{body}");
    assert!(
        body.contains("wgmma.wait_group.sync.aligned \t1;"),
        "{body}"
    );
    assert!(
        !body.contains("wgmma.wait_group.sync.aligned \t0;"),
        "{body}"
    );
    assert!(ptx.contains("wgmma.wait_group.sync.aligned \t0;"), "{ptx}");
}

#[test]
fn an_im2col_load_and_a_bulk_copy_share_a_barrier() {
    let ptx = ptx_of(crate::shared::offline_kernels::tma_im2col_load_kernel(), 90);
    assert!(
        ptx.contains("cp.async.bulk.tensor.4d.shared::cluster.global.im2col"),
        "{ptx}"
    );
    assert!(
        ptx.contains("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"),
        "{ptx}"
    );
    assert!(
        ptx.contains("mbarrier.try_wait.parity.shared::cta.b64"),
        "{ptx}"
    );
}

/// `memcpy_async` copies synchronously: on a unit barrier there is nothing to wait for, and a
/// cooperative copy splits the elements over the cube before its units meet on the `mbarrier`.
#[test]
fn barrier_copies_are_synchronous() {
    let ptx = ptx_of(crate::shared::offline_kernels::barrier_copies_kernel(), 90);
    assert_eq!(
        ptx.matches("mbarrier.try_wait").count(),
        1,
        "only the cube barrier waits:\n{ptx}"
    );
    assert!(
        !ptx.contains("cp.async"),
        "the copies are synchronous:\n{ptx}"
    );
}
