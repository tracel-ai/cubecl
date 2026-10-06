//! Real kernels for the offline tests of both GPU targets: compiled from `#[cube]` without a
//! device, so a test can assert on the instructions a lowering produces.

use cubecl_core as cubecl;
use cubecl_core::ir::{
    DeviceIdentity, HardwareProperties, MemoryDeviceProperties, features::Features,
    features::MmaConfig,
};
use cubecl_core::prelude::*;
use cubecl_runtime::kernel::CubeKernel;
use half::bf16;
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
