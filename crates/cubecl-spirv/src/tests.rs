//! Codegen assertions on the SPIR-V text for both fp8 states of a driver, without a device. The
//! IR is expanded through the CPU runtime's client and compiled with the option set by hand.

use cubecl_common::e4m3;
use cubecl_core::{
    self as cubecl, VulkanCompilationOptions, WgpuCompilationOptions, ir::settings::Dim3,
    prelude::*,
};
use cubecl_cpu::CpuRuntime;
use cubecl_runtime::{compiler::CubeTask, kernel::KernelTask};

use crate::SpirvCompiler;

#[cube(launch_unchecked)]
fn fp8_round_trip(input: &[e4m3], output: &mut [e4m3]) {
    let value = f32::cast_from(input[ABSOLUTE_POS]);
    output[ABSOLUTE_POS] = e4m3::cast_from(value * 2.0);
}

/// The disassembled SPIR-V of [`fp8_round_trip`] for a driver with or without fp8.
fn fp8_round_trip_spirv(supports_float8: bool) -> String {
    let client = CpuRuntime::client(&Default::default());
    let settings =
        KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32);
    let kernel = fp8_round_trip::Fp8RoundTrip::<CpuRuntime>::new(
        settings,
        client,
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    );
    let options = WgpuCompilationOptions {
        supports_vulkan_compiler: true,
        vulkan: VulkanCompilationOptions {
            supports_float8,
            max_spirv_version: (1, 6),
            max_vector_size: 4,
            push_constant_size: 128,
            ..Default::default()
        },
        ..Default::default()
    };
    let task = KernelTask::<SpirvCompiler, _>::new(kernel);
    let definition = task.define();
    task.compile(definition, &mut SpirvCompiler, &options)
        .expect("fp8 kernel compiles")
        .repr
        .expect("compiled SPIR-V")
        .to_string()
}

#[test]
fn with_the_extension_fp8_is_a_float_and_encoding_saturates() {
    let spirv = fp8_round_trip_spirv(true);
    assert!(spirv.contains("OpCapability Float8EXT"), "{spirv}");
    assert!(spirv.contains("OpTypeFloat 8"), "{spirv}");
    assert!(spirv.contains("OpFConvert"), "{spirv}");
    assert!(
        spirv.contains("SaturatedToLargestFloat8NormalConversionEXT"),
        "{spirv}"
    );
}

#[test]
fn without_the_extension_fp8_is_a_byte_and_no_float8_is_named() {
    let spirv = fp8_round_trip_spirv(false);
    assert!(!spirv.contains("Float8"), "{spirv}");
    assert!(spirv.contains("OpTypeInt 8"), "{spirv}");
    assert!(!spirv.contains("OpFConvert"), "{spirv}");
}
