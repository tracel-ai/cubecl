use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::config::{
    CubeClRuntimeConfig, RuntimeConfig,
    compilation::{CompilationConfig, F16Evaluation},
};
use cubecl_server::runtime::Runtime;
use std::sync::Once;

/// A client evaluating f16 as `mode`, or as the host chooses where `None`. The mode is read once,
/// when the runtime builds its device, so each mode has its own test binary.
pub fn client_evaluating(mode: Option<F16Evaluation>) -> Client {
    static CONFIG: Once = Once::new();
    CONFIG.call_once(|| {
        CubeClRuntimeConfig::set(CubeClRuntimeConfig {
            compilation: CompilationConfig {
                f16_evaluation: mode,
                ..Default::default()
            },
            ..Default::default()
        })
    });
    CpuRuntime::client(&Default::default())
}
