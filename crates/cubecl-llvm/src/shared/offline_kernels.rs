//! Real kernels for the offline tests of both GPU targets: compiled from `#[cube]` without a
//! device, so a test can assert on the instructions a lowering produces.

use cubecl_core as cubecl;
use cubecl_core::ir::{
    DeviceIdentity, HardwareProperties, MemoryDeviceProperties, features::Features,
    features::MmaConfig,
};
#[cfg(feature = "nvptx")]
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_runtime::kernel::CubeKernel;
use half::{bf16, f16};
use std::sync::Arc;

pub(crate) fn device_properties(plane_dim: u32) -> Arc<DeviceProperties> {
    let hardware = HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: plane_dim,
        plane_size_max: plane_dim,
        max_bindings: 32,
        max_shared_memory_size: 65536,
        max_cube_count: (u32::MAX, u16::MAX as u32, u16::MAX as u32),
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 1024),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        num_cpu_cores: None,
        last_level_cache_size: None,
        max_vector_size: VectorSize::MAX,
        cube_mma_reserved_shared_memory: 0,
    };
    Arc::new(DeviceProperties::new(
        Features::default(),
        MemoryDeviceProperties::new(u64::MAX, 256),
        hardware,
        cubecl_core::profile::TimingMethod::Device,
        DeviceIdentity {
            name: "offline".to_string(),
            fingerprint: "offline".to_string(),
            physical: None,
        },
    ))
}

#[cube(launch)]
fn scale(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

pub(crate) fn scale_kernel(address_type: AddressType) -> impl CubeKernel {
    let settings = KernelSettings::new(*CubeDim::new_1d(64), ExecutionMode::Checked, address_type);
    scale::Scale::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

#[cube(launch)]
fn plane_moves(input: &[f32], output: &mut [f32]) {
    let value = input[UNIT_POS as usize];
    let first = plane_broadcast(value, 0u32);
    let swapped = plane_shuffle_xor(value, 1u32);
    output[UNIT_POS as usize] = first + swapped;
}

pub(crate) fn plane_moves_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    plane_moves::PlaneMoves::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// Keeps the `K` largest values seen, in a local array updated by a loop of `K` steps per
/// input: the shape of a top-k accumulator.
// Cube code indexes: it has no iterators to lower.
#[allow(clippy::needless_range_loop)]
#[cube(launch)]
fn keep_largest(input: &[f32], output: &mut [f32], #[comptime] k: usize) {
    let mut largest = Array::<f32>::new(k);
    #[unroll]
    for i in 0..k {
        largest[i] = f32::min_value();
    }
    for r in 0..input.len() {
        let mut candidate = input[r];
        for j in 0..k {
            let keep = largest[j] > candidate;
            let displaced = select(keep, candidate, largest[j]);
            largest[j] = select(keep, largest[j], candidate);
            candidate = displaced;
        }
    }
    #[unroll]
    for i in 0..k {
        output[i] = largest[i];
    }
}

pub(crate) fn keep_largest_kernel(k: usize) -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    keep_largest::KeepLargest::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        k,
    )
}

/// How a cube waits for its turn on a relay's counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Wait {
    /// Spins on an atomic load, as cubek's `Relay::take` does.
    Load,
    /// Spins on a compare-exchange that writes back what it reads.
    CompareExchange,
}

/// One turn of a relay between cubes: wait until `turns` names this cube, add the `weights`
/// into the `carry` the turn before left, and hand the turn on. The shape of a hypercube
/// routine, whose cubes take turns at the same rows within a launch.
#[cube(launch)]
fn relay(
    turns: &[Atomic<u32>],
    carry: &mut [f32],
    weights: &[f32],
    out: &mut [f32],
    #[comptime] wait: Wait,
) {
    let turn = CUBE_POS as u32;
    if UNIT_POS_PLANE == 0 {
        loop {
            if comptime!(wait == Wait::Load) {
                if turns[0].load() == turn {
                    break;
                }
            } else if turns[0].compare_exchange_weak(turn, turn) == turn {
                break;
            }
        }
    }
    sync_storage();
    let sum = carry[0] + weights[0];
    out[ABSOLUTE_POS] = sum;
    sync_storage();
    if UNIT_POS_PLANE == 0 {
        carry[0] = sum;
        turns[0].store(turn + 1);
    }
}

pub(crate) fn relay_kernel(wait: Wait) -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    relay::Relay::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        wait,
    )
}

/// Scales `input` into `output` and counts the units into `tally`, dropping what the count was:
/// an accumulation no cube waits on.
#[cube(launch)]
fn tally(input: &[f32], output: &mut [f32], tally: &[Atomic<u32>]) {
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    tally[0].fetch_add(1u32);
}

pub(crate) fn tally_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    tally::Tally::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// Walks a column in steps of 32 rows, one row per unit: how a matmul reads its weight along
/// the reduced axis. A cube starts at its position split by a runtime count, so the start of
/// every address holds a division by a value only known at launch.
#[cube(launch)]
fn strided_walk(
    input: &[f32],
    output: &mut [f32],
    bands: u32,
    pitch: u32,
    stride: u32,
    steps: u32,
) {
    let start = (CUBE_POS_Y % bands) * pitch;
    let mut sum = 0.0f32;
    for step in 0..steps {
        let row = step * 32 + UNIT_POS_X;
        sum += input[(row * stride + start) as usize];
    }
    output[ABSOLUTE_POS] = sum;
}

pub(crate) fn strided_walk_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    strided_walk::StridedWalk::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        (),
        (),
        (),
        (),
    )
}

#[cfg(feature = "nvptx")]
#[cube(launch)]
fn tf32_round(input: &[Vector<f32, Const<4>>], output: &mut [Vector<f32, Const<4>>]) {
    if ABSOLUTE_POS < input.len() {
        let rounded = Vector::<tf32, Const<4>>::cast_from(input[ABSOLUTE_POS]);
        output[ABSOLUTE_POS] = Vector::<f32, Const<4>>::cast_from(rounded);
    }
}

#[cfg(feature = "nvptx")]
pub(crate) fn tf32_round_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Checked,
        AddressType::U32,
    );
    tf32_round::Tf32Round::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

#[cfg(feature = "nvptx")]
type Tf32 = tf32;

#[cfg(feature = "nvptx")]
#[cube(launch)]
fn tf32_round_constants(output: &mut [f32]) {
    output[0] = f32::cast_from(Tf32::cast_from(1.00048828125f32));
    output[1] = f32::cast_from(Tf32::cast_from(-1.00048828125f32));
    output[2] = f32::cast_from(Tf32::cast_from(2049i32));
    output[3] = f32::cast_from(Tf32::cast_from(4098u32));
}

#[cfg(feature = "nvptx")]
pub(crate) fn tf32_round_constants_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(1),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    tf32_round_constants::Tf32RoundConstants::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
    )
}

/// `bf16` arithmetic, a comparison, a square root and a plane reduction, which the backend
/// lowers as native `bfloat`.
#[cube(launch)]
fn bf16_math(input: &[Vector<bf16, Const<4>>], output: &mut [Vector<bf16, Const<4>>]) {
    let value = input[UNIT_POS as usize];
    let scaled = value * Vector::new(bf16::new(1.5f32)) + Vector::new(bf16::new(0.25f32));
    let positive = select_many(
        scaled.greater_than(&Vector::new(bf16::new(0.0f32))),
        scaled,
        scaled.abs().sqrt(),
    );
    output[UNIT_POS as usize] = plane_sum(positive);
}

pub(crate) fn bf16_math_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    bf16_math::Bf16Math::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// One tile product of `I` operands accumulating in `A`.
#[cube(launch)]
fn tile_product<I: Float, A: Float>(
    lhs: &[I],
    rhs: &[I],
    out: &mut [A],
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let a = cmma::Matrix::<I>::from_slice(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        lhs,
        k as u32,
    );
    let b = cmma::Matrix::<I>::from_slice(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        rhs,
        k as u32,
    );
    let c = cmma::Matrix::<A>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        A::from_int(0),
    );
    cmma::execute(&a, &b, &c, &c);
    cmma::store(out, &c, n as u32, cmma::MatrixLayout::RowMajor);
}

/// One `(m, n, k)` tile product of `I` operands accumulating in `A`.
pub(crate) fn tile_product_kernel<
    I: Float + cubecl_core::CubeElement,
    A: Float + cubecl_core::CubeElement,
>(
    (m, n, k): (usize, usize, usize),
) -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    let mut props = (*device_properties(32)).clone();
    props.features.matmul.cmma.insert(MmaConfig {
        a_type: I::cube_type(),
        b_type: I::cube_type(),
        cd_type: A::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    });
    tile_product::TileProduct::<I, A>::new(
        settings,
        Arc::new(props),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        m,
        n,
        k,
    )
}

#[cfg(feature = "nvptx")]
#[cube(launch)]
fn warpgroup_product<I: Numeric, A: Numeric>(
    lhs: &[I],
    rhs: &[I],
    out: &mut [A],
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] a_in_registers: bool,
) {
    let a_layout = comptime![wgmma::WgmmaTileLayout {
        major: wgmma::Major::K,
        swizzle: wgmma::Swizzle::None,
        rows: 64,
        k,
    }];
    // `B` is swizzled 32 bytes wide, a pattern that repeats every 256.
    let b_layout = comptime![wgmma::WgmmaTileLayout {
        major: wgmma::Major::K,
        swizzle: wgmma::Swizzle::B32,
        rows: n,
        k,
    }];
    let mut smem_a = Shared::new_aligned_slice(64 * k, 128usize);
    let mut smem_b = Shared::new_aligned_slice(n * k, 256usize);
    let unit = UNIT_POS as usize;
    if unit < 64 * k {
        smem_a[unit] = lhs[unit];
    }
    if unit < n * k {
        smem_b[unit] = rhs[unit];
    }
    sync_async_proxy_shared();
    sync_cube();

    let b = wgmma::MatrixDescriptor::new(&smem_b, b_layout);
    let mut acc = wgmma::Accumulator::<A>::new(n).start();
    if a_in_registers {
        let mut fragment = wgmma::Fragment::<I>::new();
        #[unroll]
        for nth in 0..fragment.len() {
            fragment.set(nth, lhs[unit * 16 + nth]);
        }
        acc.execute_registers(&fragment, &b);
    } else {
        let a = wgmma::MatrixDescriptor::new(&smem_a, a_layout);
        acc.execute(&a, &b);
    }
    let acc = acc.wait();

    let len = acc.len();
    #[unroll]
    for nth in 0..len {
        out[unit * len + nth] = acc.get(nth);
    }
}

/// `stages` tiles of K, one stage of MMAs committed per iteration of a loop the compiler can't
/// unroll: each iteration waits for the previous stage's group, while its own runs on.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn warpgroup_pipeline(input: &[f16], out: &mut [f32], stages: u32) {
    let layout = comptime![wgmma::WgmmaTileLayout {
        major: wgmma::Major::K,
        swizzle: wgmma::Swizzle::B128,
        rows: 64,
        k: 64,
    }];
    let mut tiles = Shared::new_aligned_slice(8192usize, 1024usize);
    let unit = UNIT_POS as usize;
    #[unroll]
    for i in 0..64usize {
        tiles[i * 128 + unit] = input[i * 128 + unit];
    }
    sync_async_proxy_shared();
    sync_cube();

    let mut acc = wgmma::Accumulator::<f32>::new(64usize).start();
    let mut released = acc.commit();
    for stage in 0..stages {
        let at = (stage % 2) as usize * 4096;
        let a = wgmma::MatrixDescriptor::new(&tiles[at..at + 4096], layout);
        let b = wgmma::MatrixDescriptor::new(&tiles[at..at + 4096], layout);
        #[unroll]
        for step in 0..4usize {
            acc.execute(&a.at(0usize, step * 16), &b.at(0usize, step * 16));
        }
        let read = acc.commit();
        released.wait();
        released = read;
    }
    let acc = acc.wait();
    out[unit] = acc.get(0usize);
}

#[cfg(feature = "nvptx")]
pub(crate) fn warpgroup_pipeline_kernel() -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(128),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    let mut props = (*device_properties(32)).clone();
    props
        .features
        .matmul
        .wgmma
        .extend(crate::nvptx::wgmma::wgmma_configs().iter().copied());
    warpgroup_pipeline::WarpgroupPipeline::new(
        settings,
        Arc::new(props),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        (),
    )
}

/// One warpgroup MMA of `I` operands accumulating in `A`, `64 x n x k`.
#[cfg(feature = "nvptx")]
pub(crate) fn warpgroup_product_kernel<
    I: Numeric + cubecl_core::CubeElement,
    A: Numeric + cubecl_core::CubeElement,
>(
    n: usize,
    k: usize,
    a_in_registers: bool,
) -> impl CubeKernel {
    let settings = KernelSettings::new(
        *CubeDim::new_1d(128),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    let mut props = (*device_properties(32)).clone();
    props
        .features
        .matmul
        .wgmma
        .insert(cubecl_core::ir::features::WgmmaConfig {
            a_type: I::cube_type(),
            b_type: I::cube_type(),
            cd_type: A::cube_type(),
            m: 64,
            n_granularity: n as u32,
            n_max: n as u32,
            k: k as u32,
        });
    warpgroup_product::WarpgroupProduct::<I, A>::new(
        settings,
        Arc::new(props),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
        n,
        k,
        a_in_registers,
    )
}

/// Device properties of a Hopper with TMA, for kernels that use it.
#[cfg(feature = "nvptx")]
fn tma_properties() -> Arc<DeviceProperties> {
    use cubecl_core::ir::{OpaqueType, features::Tma};

    let mut props = (*device_properties(32)).clone();
    props.features.tma.insert(Tma::Base);
    props.register_opaque_type(OpaqueType::TensorMap);
    props.register_opaque_type(OpaqueType::Barrier);
    Arc::new(props)
}

#[cfg(feature = "nvptx")]
fn tma_settings() -> KernelSettings {
    KernelSettings::new(
        *CubeDim::new_2d(32, 16),
        ExecutionMode::Unchecked,
        AddressType::U32,
    )
}

/// A tile loaded by TMA, which unit 0 issues and every unit waits on.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn tma_tile_load(input: &TensorMap<f32, Tiled>, output: &mut [f32]) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = Shared::<[f32]>::new_aligned_slice(32usize * 16, 128usize);

    let expected = select(UNIT_POS == 0, 32u32 * 16 * 4, 0);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, stage.as_mut_slice(), 0, 8);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    output[UNIT_POS as usize] = stage[UNIT_POS as usize];
}

#[cfg(feature = "nvptx")]
pub(crate) fn tma_tile_load_kernel() -> impl CubeKernel {
    tma_tile_load::TmaTileLoad::new(
        tma_settings(),
        tma_properties(),
        Arc::new(TargetProperties::default()),
        (),
        BufferCompilationArg { inplace: None },
    )
}

/// A tile stored by TMA from shared memory the units wrote.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn tma_tile_store(input: &[f32], output: &mut TensorMap<f32, Tiled>) {
    let mut stage = Shared::<[f32]>::new_aligned_slice(32usize * 16, 128usize);
    stage[UNIT_POS as usize] = input[UNIT_POS as usize];
    sync_async_proxy_shared();
    sync_cube();

    if UNIT_POS == 0 {
        // Two stores in flight: the first is done reading once at most one group is left.
        let first = tma_store_2d(&stage[0..256], output, 0, 0);
        let second = tma_store_2d(&stage[256..512], output, 16, 8);
        first.wait();
        second.wait();
    }
}

#[cfg(feature = "nvptx")]
pub(crate) fn tma_tile_store_kernel() -> impl CubeKernel {
    tma_tile_store::TmaTileStore::new(
        tma_settings(),
        tma_properties(),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        (),
    )
}

/// An im2col load, waited on by phase, next to a one-dimensional bulk copy on the same barrier.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn tma_im2col_load(input: &TensorMap<f32, Im2col>, bias: &[f32], output: &mut [f32]) {
    let barrier = Barrier::shared(1u32, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = Shared::<[f32]>::new_aligned_slice(32usize * 16, 128usize);
    let mut stage_bias = Shared::<[f32]>::new_aligned_slice(32usize, 128usize);

    if UNIT_POS == 0 {
        barrier.tma_load_im2col_4d(input, stage.as_mut_slice(), 0, -1, -1, 0, 1u16, 2u16);
        barrier.memcpy_async_tx(&bias[0..32], stage_bias.as_mut_slice());
        barrier.arrive_and_expect_tx(1, 32u32 * 16 * 4 + 32 * 4);
    }
    barrier.wait_parity(0);

    output[UNIT_POS as usize] = stage[UNIT_POS as usize] + stage_bias[(UNIT_POS % 32) as usize];
}

#[cfg(feature = "nvptx")]
pub(crate) fn tma_im2col_load_kernel() -> impl CubeKernel {
    tma_im2col_load::TmaIm2colLoad::new(
        tma_settings(),
        tma_properties(),
        Arc::new(TargetProperties::default()),
        (),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// `memcpy_async` on a unit barrier, then a cooperative one on a cube barrier.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn barrier_copies(input: &[f32], output: &mut [f32]) {
    let mut own = Shared::<[f32]>::new_aligned_slice(512usize, 16usize);
    let mut all = Shared::<[f32]>::new_aligned_slice(512usize, 16usize);
    let unit = UNIT_POS as usize;

    let local = Barrier::local();
    local.memcpy_async(&input[unit..unit + 1], &mut own[unit..unit + 1]);
    local.arrive_and_wait();

    let shared = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    shared.memcpy_async_cooperative(&input[0..512], all.as_mut_slice());
    shared.arrive_and_wait();

    output[unit] = own[unit] + all[511 - unit];
}

#[cfg(feature = "nvptx")]
pub(crate) fn barrier_copies_kernel() -> impl CubeKernel {
    barrier_copies::BarrierCopies::new(
        tma_settings(),
        tma_properties(),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// How a kernel of [`group_waits`] commits and waits.
#[cfg(feature = "nvptx")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WaitCase {
    /// A store on one branch only, between the first store and its wait.
    OneBranchStores,
    /// A store on both branches.
    BothBranchesStore,
    /// Two stores per iteration, the wait on the last store of the previous one.
    TwoStoresPerIteration,
    /// The stores between a wait and the token it waits on are in a loop that may not run.
    MaybeEmptyInnerLoop,
    /// A warpgroup MMA group committed after the store, which the store's wait doesn't count.
    OtherKind,
}

/// Bulk stores of a tile, committed and waited on as `case` says.
#[cfg(feature = "nvptx")]
#[cube(launch)]
fn group_waits(output: &mut TensorMap<f32, Tiled>, count: u32, #[comptime] case: WaitCase) {
    let stage = Shared::<[f32]>::new_aligned_slice(32usize * 16, 128usize);
    let tile = stage.as_slice();
    match comptime![case] {
        WaitCase::OneBranchStores => {
            let first = tma_store_2d(tile, output, 0, 0);
            if count > 1 {
                tma_store_2d(tile, output, 16, 0);
            }
            first.wait();
        }
        WaitCase::BothBranchesStore => {
            let first = tma_store_2d(tile, output, 0, 0);
            if count > 1 {
                tma_store_2d(tile, output, 16, 0);
            } else {
                tma_store_2d(tile, output, 32, 0);
            }
            first.wait();
        }
        WaitCase::TwoStoresPerIteration => {
            let mut previous = tma_store_2d(tile, output, 0, 0);
            for i in 0..count {
                tma_store_2d(tile, output, i as i32, 0);
                let second = tma_store_2d(tile, output, i as i32, 8);
                previous.wait();
                previous = second;
            }
        }
        WaitCase::MaybeEmptyInnerLoop => {
            let mut previous = tma_store_2d(tile, output, 0, 0);
            for i in 0..count {
                for j in 0..i {
                    tma_store_2d(tile, output, j as i32, 0);
                }
                previous.wait();
                previous = tma_store_2d(tile, output, i as i32, 8);
            }
        }
        WaitCase::OtherKind => {
            let stored = tma_store_2d(tile, output, 0, 0);
            let mut acc = wgmma::Accumulator::<f32>::new(8usize).start();
            acc.commit();
            stored.wait();
            let _ = acc.wait();
        }
    }
}

#[cfg(feature = "nvptx")]
pub(crate) fn group_waits_kernel(case: WaitCase) -> impl CubeKernel {
    group_waits::GroupWaits::new(
        KernelSettings::new(
            *CubeDim::new_1d(128),
            ExecutionMode::Unchecked,
            AddressType::U32,
        ),
        tma_properties(),
        Arc::new(TargetProperties::default()),
        (),
        (),
        case,
    )
}
