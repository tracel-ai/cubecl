#![cfg(any(target_os = "linux", target_os = "windows"))]

//! Alone in its binary: the configuration is loaded once per process, and clearing every
//! throughput cache would make another test probe again.

use cubecl_core::throughput::ThroughputCache;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use std::time::{Duration, Instant};

/// Far below what one probe takes, which spends a quarter of a second on its plateau alone.
const CACHED_ANSWER_LIMIT: Duration = Duration::from_millis(50);

#[test]
fn with_the_cache_off_clear_all_makes_a_key_probe_again() {
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("CUBECL_THROUGHPUT_CACHE", "off") };
    let client = HipRuntime::client(&Default::default());
    cubecl_std::throughput::measure_launch_overhead(&client);

    ThroughputCache::clear_all();
    let start = Instant::now();
    cubecl_std::throughput::measure_launch_overhead(&client);
    let elapsed = start.elapsed();

    assert!(
        elapsed >= CACHED_ANSWER_LIMIT,
        "answered from the cache, in {elapsed:?}"
    );
}
