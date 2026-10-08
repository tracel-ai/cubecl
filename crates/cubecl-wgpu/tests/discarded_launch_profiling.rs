//! A discarded launch under the profiling logger, which times every launch.
//!
//! wgpu times a window with the timestamps its compute passes write, and a launch an execution
//! override drops opens no pass.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel};
use cubecl_server::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

#[test]
fn a_discarded_launch_is_issued_under_the_profiling_logger() {
    let mut config = CubeClRuntimeConfig::default();
    config.profiling.logger.level = ProfilingLogLevel::Medium;
    CubeClRuntimeConfig::set(config);

    let client = <WgpuRuntime>::client(&Default::default());
    let out = client.empty(core::mem::size_of::<u32>());

    let execution =
        ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &StatisticsCollector::new());
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
    );
    core::mem::drop(execution);

    client
        .read_one(out)
        .expect("a discarded launch fails nothing it would have written");
}
