//! A service asked to close at exit runs the work queued on it before the process ends.
//!
//! The test runs itself again in a child process that queues slow work and returns from the
//! test without waiting for it, so the process exits with the work still queued.

use cubecl_common::device::{DeviceId, DeviceService, ServerUtilitiesHandle};
use cubecl_common::device_handle::DeviceHandle;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

/// Where the child writes once its queued work ran, set only in the child.
const MARKER: &str = "CUBECL_TEST_EXIT_DRAIN_MARKER";

struct SlowService;

impl DeviceService for SlowService {
    fn init(_device_id: DeviceId) -> Self {
        Self
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
}

#[test]
fn queued_work_runs_before_exit() {
    match std::env::var_os(MARKER) {
        Some(marker) => queue_slow_work_and_return(PathBuf::from(marker)),
        None => run_child(),
    }
}

fn queue_slow_work_and_return(marker: PathBuf) {
    let device_id = DeviceId {
        type_id: 0,
        index_id: 0,
    };
    let handle = DeviceHandle::<SlowService>::new(device_id);
    DeviceHandle::<SlowService>::close_at_exit(device_id);
    handle.submit(move |_| {
        std::thread::sleep(Duration::from_millis(300));
        std::fs::write(marker, "ran").unwrap();
    });
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
