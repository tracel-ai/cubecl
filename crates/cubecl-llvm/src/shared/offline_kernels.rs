//! Real kernels for the offline tests of both GPU targets: compiled from `#[cube]` without a
//! device, so a test can assert on the instructions a lowering produces.

use cubecl_core as cubecl;
use cubecl_core::ir::{
    DeviceIdentity, HardwareProperties, MemoryDeviceProperties, features::Features,
};
use cubecl_core::prelude::*;
use cubecl_runtime::kernel::CubeKernel;
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
