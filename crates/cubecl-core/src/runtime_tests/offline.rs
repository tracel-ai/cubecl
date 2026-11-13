//! A device and kernels for compiler tests that run without one.

use alloc::{string::ToString, sync::Arc};
use cubecl_ir::{
    DeviceIdentity, DeviceProperties, HardwareProperties, MemoryDeviceProperties, VectorSize,
    features::Features, settings::DebugInfo,
};
use cubecl_runtime::kernel::CubeKernel;

use crate as cubecl;
use crate::prelude::*;
use crate::profile::TimingMethod;

/// The properties of a device with planes of `plane_dim` units and no type registered. A compiler
/// test registers the types its target supports.
#[must_use]
pub fn offline_device_properties(plane_dim: u32) -> DeviceProperties {
    let hardware = HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: plane_dim,
        plane_size_max: plane_dim,
        max_bindings: 32,
        max_shared_memory_size: 65536,
        max_cube_count: (u32::MAX, u32::from(u16::MAX), u32::from(u16::MAX)),
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
    DeviceProperties::new(
        Features::default(),
        MemoryDeviceProperties::new(u64::MAX, 256),
        hardware,
        TimingMethod::Device,
        DeviceIdentity {
            name: "offline".to_string(),
            fingerprint: "offline".to_string(),
            physical: None,
        },
    )
}

// The debug data tests of the compilers find the lines below with `source_line`, so each
// statement occurs one time in this file.

#[cube]
#[must_use]
pub fn square_third(x: f32) -> f32 {
    let y = x * x;
    y / 3.0
}

#[cube]
#[must_use]
pub fn doubled(x: f32) -> f32 {
    let third = square_third(x);
    third * 2.0
}

#[cube(launch)]
pub fn nested_calls(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = doubled(input[ABSOLUTE_POS]);
    }
}

/// `nested_calls` with `debug_symbols`: the macro records the text of this file.
#[cube(launch, debug_symbols)]
pub fn nested_calls_with_source(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = doubled(input[ABSOLUTE_POS]) + 1.0;
    }
}

/// The text of this file, as `nested_calls_with_source` records it.
pub const SOURCE: &str = include_str!("offline.rs");

/// The path of this file in the debug data, relative to the workspace root.
pub const SOURCE_PATH: &str = "crates/cubecl-core/src/runtime_tests/offline.rs";

/// The line of this file that contains `text`.
///
/// # Panics
/// When no line or more than one line contains `text`.
#[must_use]
pub fn source_line(text: &str) -> u32 {
    let mut lines = SOURCE
        .lines()
        .zip(1..)
        .filter(|(line, _)| line.contains(text));
    let (_, number) = lines
        .next()
        .unwrap_or_else(|| panic!("no line has `{text}`"));
    assert!(lines.next().is_none(), "more than one line has `{text}`");
    number
}

fn settings(level: DebugInfo) -> KernelSettings {
    KernelSettings::new(
        *CubeDim::new_1d(64),
        ExecutionMode::Checked,
        AddressType::U32,
    )
    .debug_info(level)
}

/// `nested_calls → doubled → square_third` at `level`, on a device with `properties`.
#[must_use]
pub fn nested_calls_kernel(properties: Arc<DeviceProperties>, level: DebugInfo) -> impl CubeKernel {
    nested_calls::NestedCalls::new(
        settings(level),
        properties,
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

/// `nested_calls_with_source` at [`DebugInfo::Full`], on a device with `properties`.
#[must_use]
pub fn nested_calls_with_source_kernel(properties: Arc<DeviceProperties>) -> impl CubeKernel {
    nested_calls_with_source::NestedCallsWithSource::new(
        settings(DebugInfo::Full),
        properties,
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}
