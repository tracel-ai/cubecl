//! Real kernels compiled for AMDGPU without a device, checked on the assembly.

use crate::target::LlvmTarget;
use crate::{PlironArtifact, PlironCompiler, PlironOptions, amdgpu::codegen::compile_to_object};
use cubecl_core as cubecl;
use cubecl_core::ir::amd::GfxArch;
use cubecl_core::ir::{
    DeviceIdentity, HardwareProperties, MemoryDeviceProperties, features::Features,
};
use cubecl_core::{Compiler, prelude::*};
use cubecl_runtime::kernel::CubeKernel;
use std::sync::Arc;

fn device_properties(plane_dim: u32) -> Arc<DeviceProperties> {
    let hardware = HardwareProperties {
        load_width: 128,
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
    compile_to_object(&module.ir, &arch, true)
        .unwrap()
        .1
        .unwrap()
}

#[cube(launch)]
fn scale(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

fn scale_kernel(address_type: AddressType) -> impl CubeKernel {
    let settings = KernelSettings::new(*CubeDim::new_1d(64), ExecutionMode::Checked, address_type);
    scale::Scale::new(
        settings,
        device_properties(32),
        Arc::new(TargetProperties::default()),
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    )
}

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

#[cube(launch)]
fn plane_moves(input: &[f32], output: &mut [f32]) {
    let value = input[UNIT_POS as usize];
    let first = plane_broadcast(value, 0u32);
    let swapped = plane_shuffle_xor(value, 1u32);
    output[UNIT_POS as usize] = first + swapped;
}

fn plane_moves_kernel() -> impl CubeKernel {
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
