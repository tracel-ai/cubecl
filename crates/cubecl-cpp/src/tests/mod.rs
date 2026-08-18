//! Codegen assertions on the generated C++ text: which fp8 path each dialect emits, without a
//! device. The IR is expanded through the CPU runtime's client and printed by each dialect.

use cubecl_common::e4m3;
use cubecl_core::{self as cubecl, ir::settings::Dim3, prelude::*};
use cubecl_cpu::CpuRuntime;
use cubecl_runtime::{
    compiler::{Compiler, CubeTask},
    kernel::KernelTask,
};

use crate::{
    shared::{CompilationOptions, CppCompiler},
    target::{CppTarget, Cuda, Hip},
};

#[cube(launch_unchecked)]
fn fp8_round_trip(input: &[e4m3], output: &mut [e4m3]) {
    let value = f32::cast_from(input[ABSOLUTE_POS]);
    output[ABSOLUTE_POS] = e4m3::cast_from(value * 2.0);
}

/// The C++ a target's dialect prints for [`fp8_round_trip`].
fn fp8_round_trip_source<T: CppTarget>() -> String
where
    CppCompiler<T>: Compiler<CompilationOptions = CompilationOptions>,
{
    let client = CpuRuntime::client(&Default::default());
    let settings =
        KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32);
    let kernel = fp8_round_trip::Fp8RoundTrip::<CpuRuntime>::new(
        settings,
        client,
        BufferCompilationArg { inplace: None },
        BufferCompilationArg { inplace: None },
    );
    let task = KernelTask::<CppCompiler<T>, _>::new(kernel);
    let definition = task.define();
    task.compile(
        definition,
        &mut CppCompiler::<T>::default(),
        &CompilationOptions::default(),
    )
    .expect("fp8 kernel compiles")
    .source
}

#[test]
fn cuda_converts_fp8_with_the_header_intrinsics() {
    let source = fp8_round_trip_source::<Cuda>();
    assert!(source.contains("cuda_fp8.h"), "{source}");
    assert!(source.contains("__nv_cvt_fp8_to_halfraw"), "{source}");
    // Straight from f32 and saturating: the f16 detour rounds twice and NOSAT is the header's
    // software path.
    assert!(
        source.contains("__nv_cvt_float_to_fp8(") && source.contains("__NV_SATFINITE"),
        "{source}"
    );
    assert!(!source.contains("__NV_NOSAT"), "{source}");
    assert!(!source.contains("__nv_cvt_halfraw_to_fp8"), "{source}");
}

#[test]
fn hip_converts_fp8_in_software_on_bytes() {
    let source = fp8_round_trip_source::<Hip>();
    assert_software_fp8(&source);
}

#[cfg(feature = "metal")]
#[test]
fn metal_converts_fp8_in_software_on_bytes() {
    let source = fp8_round_trip_source::<crate::target::Metal>();
    assert_software_fp8(&source);
}

/// The fp8 buffers are plain bytes and no vendor conversion intrinsic or header is named.
fn assert_software_fp8(source: &str) {
    assert!(source.contains("uint8_t const*"), "{source}");
    for vendor in ["__nv_", "cuda_fp8", "__hip_fp8", "hip_fp8"] {
        assert!(!source.contains(vendor), "{vendor}: {source}");
    }
}
