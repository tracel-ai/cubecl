use core::sync::atomic::AtomicU64;

/// A unique id for a transfer from one compute server to another
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeDataTransferId(u64);

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl ComputeDataTransferId {
    /// Get a new unique transfer id. 
    pub fn new() -> Self {
        let val = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Self(val)
    }
}

