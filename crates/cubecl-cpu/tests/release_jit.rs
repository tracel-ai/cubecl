//! A release build gives kernels no debug data, and the JIT then has the default settings of LLVM:
//! `JITLink`, without the gdb listener. The tests build with debug data, so the configuration
//! limit gives this test the level of a release build.
//!
//! The configuration is read once for each process, so this file has one test.

use cubecl_core::ir::settings::DebugInfo;
use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::offline::nested_calls;
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_server::config::{
    CubeClRuntimeConfig, RuntimeConfig, compilation::effective_debug_info,
};
use cubecl_server::runtime::Runtime;

#[test]
fn a_kernel_without_debug_data_runs() {
    let mut config = CubeClRuntimeConfig::default();
    config.compilation.debug_info = Some(DebugInfo::None);
    CubeClRuntimeConfig::set(config);
    assert_eq!(effective_debug_info(DebugInfo::Full), DebugInfo::None);

    let client = CpuRuntime::client(&CpuDevice);
    let input = client.create_from_slice(f32::as_bytes(&[3.0, 1.5, 0.0, -6.0]));
    let output = client.empty(4 * size_of::<f32>());
    // SAFETY: each handle holds 4 values of `f32`.
    let buffer = |handle| unsafe { BufferArg::from_raw_parts(handle, 4) };
    nested_calls::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(4),
        buffer(input),
        buffer(output.clone()),
    );

    let bytes = client.read_one(output).unwrap();
    // `x * x / 3 * 2`.
    assert_eq!(f32::from_bytes(&bytes), [6.0, 1.5, 0.0, 24.0]);
}
