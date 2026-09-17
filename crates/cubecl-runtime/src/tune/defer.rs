use cubecl_environment::sync::{AtomicBool, Ordering};

/// Whether a cache miss runs the plan's first working candidate instead of a
/// tuning round. A process-wide switch, read on every miss, so a caller can
/// keep a latency-sensitive path from tuning and leave the round to a call it
/// makes when the device has time — nothing is cached until one does.
static DEFERRED: AtomicBool = AtomicBool::new(false);

/// Defer tuning on a miss (`true`) or let a miss tune inline (`false`).
pub fn defer_tuning(on: bool) {
    DEFERRED.store(on, Ordering::Relaxed);
}

/// Whether a miss currently serves a default rather than tuning.
pub fn tuning_deferred() -> bool {
    DEFERRED.load(Ordering::Relaxed)
}
