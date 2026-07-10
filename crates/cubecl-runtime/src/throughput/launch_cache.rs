use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use hashbrown::HashMap;
use spin::Mutex;

/// In-memory memo of the launch overhead (a fixed per-device latency) keyed by device name.
///
/// Unlike the [`ThroughputCache`](super::ThroughputCache) this is process-local only:
/// launch overhead is cheap to measure and is a latency rather than a throughput, so it
/// doesn't fit the on-disk [`ThroughputValue`](super::ThroughputValue) schema.
static LAUNCH_OVERHEAD_CACHE: Mutex<Option<HashMap<String, Duration>>> = Mutex::new(None);

/// Returns the cached launch overhead for `name`, measuring it on a miss.
///
/// `sample` performs a single trivial dispatch and returns its measured duration; it is
/// called repeatedly here to warm up and to gather the sample used for the estimate.
///
/// The lock is not held while sampling, so a concurrent first call for the same device
/// may sample twice; the first value stored wins and both callers observe it.
pub fn launch_overhead_or_measure(name: &str, sample: impl Fn() -> Duration) -> Duration {
    {
        let guard = LAUNCH_OVERHEAD_CACHE.lock();
        if let Some(value) = guard.as_ref().and_then(|map| map.get(name)) {
            return *value;
        }
    }

    let value = measure_overhead(sample);

    let mut guard = LAUNCH_OVERHEAD_CACHE.lock();
    *guard
        .get_or_insert_with(HashMap::new)
        .entry(name.to_string())
        .or_insert(value)
}

/// Warms up, samples the single-dispatch `sample`, and reduces to a robust estimate.
///
/// Warmup runs are discarded to exclude one-time kernel compilation. The median of the
/// timed runs is returned to stay robust to scheduler jitter and to the bogus near-zero
/// device timestamps some drivers report.
fn measure_overhead(sample: impl Fn() -> Duration) -> Duration {
    const WARMUP: usize = 3;
    const SAMPLES: usize = 20;

    for _ in 0..WARMUP {
        let _ = sample();
    }

    let mut durations: Vec<Duration> = (0..SAMPLES).map(|_| sample()).collect();
    durations.sort_unstable();

    durations[durations.len() / 2]
}
