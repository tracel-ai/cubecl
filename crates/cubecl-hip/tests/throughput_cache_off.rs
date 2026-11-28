#![cfg(any(target_os = "linux", target_os = "windows"))]

//! Alone in its binary: the configuration is loaded once per process, and every other test
//! expects the throughput cache its configuration gives.

use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use std::time::{Duration, Instant};

/// Far below what one probe takes, which spends a quarter of a second on its plateau alone.
const CACHED_ANSWER_LIMIT: Duration = Duration::from_millis(50);

#[test]
fn with_the_cache_off_a_key_is_probed_once_a_process() {
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("CUBECL_THROUGHPUT_CACHE", "off") };
    let client = HipRuntime::client(&Default::default());

    let probed = cubecl_std::throughput::measure_launch_overhead(&client);
    let start = Instant::now();
    let answered = cubecl_std::throughput::measure_launch_overhead(&client);
    let elapsed = start.elapsed();

    assert_eq!(answered, probed);
    assert!(
        elapsed < CACHED_ANSWER_LIMIT,
        "probed again, in {elapsed:?}"
    );
}
