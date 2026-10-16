//! `disabled_fast_math = ["allow-contraction"]`, which forbids contraction in every kernel,
//! including one that never mentions fast math. WGSL cannot forbid it, so only SPIR-V is tested.
#![cfg(feature = "spirv")]

use cubecl_core::ir::DisabledFastMath;
use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::fast_math;
use cubecl_server::config::{CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationConfig};
use cubecl_server::runtime::Runtime;

type WgpuRuntime = cubecl_wgpu::WgpuRuntime;

#[test]
fn disabled_contraction_reaches_every_kernel() {
    CubeClRuntimeConfig::set(CubeClRuntimeConfig {
        compilation: CompilationConfig {
            disabled_fast_math: DisabledFastMath::new(FastMath::AllowContraction.into()),
            ..Default::default()
        },
        ..Default::default()
    });
    let client = WgpuRuntime::client(&Default::default());
    fast_math::test_disabled_contraction_reaches_every_kernel::<WgpuRuntime>(client);
}
