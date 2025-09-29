use core::sync::atomic::AtomicU64;

/// A unique id for a transfer from one compute server to another
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataTransferId(u64);

impl core::fmt::Display for DataTransferId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("DataTransferId({})", self.0))
    }
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl Default for DataTransferId {
    fn default() -> Self {
        Self::new()
    }
}

impl DataTransferId {
    /// Get a new unique transfer id.
    pub fn new() -> Self {
        let val = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Self(val)
    }
}
