//! A client's server runs the work queued on it before the process exits.
//!
//! The test runs itself again in a child process that launches a slow kernel through a client
//! and returns without waiting for it, so the process exits with the work still queued. The
//! child never asks for the close itself: loading the client is what has to ask.

// Shared with the integration tests, most of which this test does not use.
#[allow(dead_code)]
mod dummy;

use crate::dummy::{DummyDevice, DummyKernel, KernelTask, test_client};
use cubecl_server::id::KernelId;
use cubecl_server::server::{CubeCount, KernelArguments};
use cubecl_server::storage::BytesResource;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Where the child writes once its queued kernel ran, set only in the child.
const MARKER: &str = "CUBECL_TEST_EXIT_DRAIN_MARKER";

/// Sleeps, then writes the marker.
#[derive(Debug)]
struct SlowKernel {
    marker: PathBuf,
}

impl DummyKernel for SlowKernel {
    fn compute(&self, _resources: &mut [&mut BytesResource]) {
        std::thread::sleep(Duration::from_millis(300));
        std::fs::write(&self.marker, "ran").unwrap();
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[test]
fn queued_work_runs_before_exit() {
    match std::env::var_os(MARKER) {
        Some(marker) => launch_slow_kernel_and_return(PathBuf::from(marker)),
        None => run_child(),
    }
}

fn launch_slow_kernel_and_return(marker: PathBuf) {
    let client = test_client(&DummyDevice);
    let buffer = client.empty(4);
    client.launch(
        Box::new(KernelTask::new(SlowKernel { marker })),
        CubeCount::Static(1, 1, 1),
        KernelArguments::new().with_buffers(vec![buffer.binding()]),
    );
}

fn run_child() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("ran");
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "queued_work_runs_before_exit",
            "--exact",
            "--test-threads=1",
        ])
        .env(MARKER, &marker)
        .status()
        .unwrap();

    assert!(status.success(), "the child exited with {status}");
    assert!(
        marker.exists(),
        "the work queued before exit never ran: the process ended first"
    );
}
