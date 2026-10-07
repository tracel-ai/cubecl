//! A dry run under the profiling logger, which times every launch.
//!
//! wgpu times a window with the timestamps its compute passes write, and a launch the dry run
//! drops opens no pass.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel};
use cubecl_server::dry_run::{DryRun, DryRunScope};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

#[test]
fn a_dry_run_launches_under_the_profiling_logger() {
    let mut config = CubeClRuntimeConfig::default();
    config.profiling.logger.level = ProfilingLogLevel::Medium;
    CubeClRuntimeConfig::set(config);

    let client = <WgpuRuntime>::client(&Default::default());
    let out = client.empty(core::mem::size_of::<u32>());

    let dry_run = DryRun::new().pass(DryRunScope::Profile);
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
    );
    drop(dry_run);

    client
        .read_one(out)
        .expect("a dropped launch fails nothing it would have written");
}
