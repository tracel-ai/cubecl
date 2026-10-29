//! A process exits cleanly while another thread still uses the device.
//!
//! The HIP runtime tears itself down in its library's destructors, which run at
//! exit while the process's other threads keep running. A server call that
//! reaches HIP after that point, such as the pinned host free a memory cleanup
//! makes, touches the freed runtime and crashes the process after its work is
//! done. A remote server's session closing as its client exits does exactly
//! that.
//!
//! The process has to exit for the race to exist, so the test runs itself again
//! as a child: the child keeps a thread reading (which stages through pinned
//! host memory) and cleaning up while its test returns and the process exits,
//! and the parent checks how the child ended.

use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use std::process::Command;
use std::time::Duration;

/// Set in the child's environment: the run that races the exit.
const CHILD: &str = "CUBECL_HIP_EXIT_TEARDOWN_CHILD";

/// What the child's busy thread prints once a read went through, so the parent
/// knows the race it checks was on.
const REACHED: &str = "the busy thread reached the device";

const TEST: &str = "a_process_exits_cleanly_while_another_thread_uses_the_device";

#[test]
fn a_process_exits_cleanly_while_another_thread_uses_the_device() {
    if !cubecl_hip_sys::is_available() {
        eprintln!("skipped: no HIP runtime on this machine");
        return;
    }
    if std::env::var_os(CHILD).is_some() {
        return race_the_exit();
    }
    // Several exits, since one may happen to fall between two calls.
    for attempt in 0..5 {
        let output = Command::new(std::env::current_exe().expect("the test binary's path"))
            .args(["--exact", TEST, "--test-threads=1", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .expect("the child runs");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "exit {attempt} ended with {} while a thread still used the device:\n{stderr}",
            output.status
        );
        assert!(
            stderr.contains(REACHED),
            "exit {attempt}: the busy thread never reached the device:\n{stderr}"
        );
        assert!(
            !stderr.contains("panicked"),
            "exit {attempt}: a thread panicked instead of being refused:\n{stderr}"
        );
    }
}

/// Leaves a thread reading and cleaning up when the test returns, which is when
/// the process exits. Calls refused once the device is closed end the thread; a
/// panic or a crash fails the parent.
fn race_the_exit() {
    let client = HipRuntime::client(&Default::default());
    let busy = client.clone();
    std::thread::spawn(move || {
        let mut reached = false;
        loop {
            let handle = busy.create_from_slice(&[7u8; 1 << 16]);
            if busy.read_one(handle).is_err() || busy.memory_cleanup().is_err() {
                return;
            }
            if !reached {
                eprintln!("{REACHED}");
                reached = true;
            }
        }
    });
    std::thread::sleep(Duration::from_millis(200));
}
