//! A discarded launch while an observer times every launch.
//!
//! wgpu times a window with the timestamps its compute passes write, and a launch an execution
//! override drops opens no pass, so it is not timed.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};
use cubecl_server::logging::{
    Execution, ExecutionObservation, ExecutionObserver, Measurement, TimedExecutions, TimingRequest,
};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

/// Asks for every launch's measurement unread, which it is handed for every timed launch, and
/// counts them.
#[derive(Default)]
struct CountingExecutionObserver {
    measurements: AtomicUsize,
}

impl ExecutionObserver for CountingExecutionObserver {
    fn observe(&self, _execution: &Execution) {}
    fn timing(&self) -> TimingRequest {
        TimingRequest::Deferred(TimedExecutions::Kernels)
    }
    fn observe_measurement(&self, _execution: &Execution, _measurement: Measurement) {
        self.measurements.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn a_discarded_launch_is_not_timed() {
    let client = <WgpuRuntime>::client(&Default::default());
    let out = client.empty(core::mem::size_of::<u32>());
    let observer = Arc::new(CountingExecutionObserver::default());

    let watching = ExecutionObservation::new(observer.clone());
    let execution =
        ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &StatisticsCollector::new());
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
    );
    core::mem::drop(execution);
    core::mem::drop(watching);

    assert_eq!(observer.measurements.load(Ordering::Relaxed), 0);
}
