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

/// The directory the child writes its markers in, set only in the child.
const MARKER_DIR: &str = "CUBECL_TEST_EXIT_DRAIN_MARKER_DIR";

/// Written by the child once it queued the kernel, before it returns.
const LAUNCHED_MARKER: &str = "launched";

/// Written by the kernel once it ran.
const RAN_MARKER: &str = "ran";

/// How long the kernel runs, long enough to still be queued when the child returns.
const KERNEL_DURATION: Duration = Duration::from_millis(300);

/// The bytes of the buffer the kernel is launched on.
const BUFFER_SIZE_BYTES: usize = 4;

/// Sleeps, then writes the marker.
#[derive(Debug)]
struct SlowKernel {
    marker: PathBuf,
}

impl DummyKernel for SlowKernel {
    fn compute(&self, _resources: &mut [&mut BytesResource]) {
        std::thread::sleep(KERNEL_DURATION);
        std::fs::write(&self.marker, RAN_MARKER).unwrap();
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[test]
fn queued_work_runs_before_exit() {
    match std::env::var_os(MARKER_DIR) {
        Some(dir) => launch_slow_kernel_and_return(PathBuf::from(dir)),
        None => run_child(),
    }
}

fn launch_slow_kernel_and_return(dir: PathBuf) {
    let client = test_client(&DummyDevice);
    let buffer = client.empty(BUFFER_SIZE_BYTES);
    client.launch(
        Box::new(KernelTask::new(SlowKernel {
            marker: dir.join(RAN_MARKER),
        })),
        CubeCount::Static(1, 1, 1),
        KernelArguments::new().with_buffers(vec![buffer.binding()]),
    );
    std::fs::write(dir.join(LAUNCHED_MARKER), LAUNCHED_MARKER).unwrap();
}

fn run_child() {
    let dir = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "queued_work_runs_before_exit",
            "--exact",
            "--test-threads=1",
        ])
        .env(MARKER_DIR, dir.path())
        .status()
        .unwrap();

    assert!(status.success(), "the child exited with {status}");
    assert!(
        dir.path().join(LAUNCHED_MARKER).exists(),
        "the child never launched the kernel: its test filter matched no test"
    );
    assert!(
        dir.path().join(RAN_MARKER).exists(),
        "the work queued before exit never ran: the process ended first"
    );
}
