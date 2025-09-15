#[cfg(feature = "std")]
use core::sync::atomic::AtomicU64;

/// Unique identifier that can represent a stream based on the current thread id.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct StreamId {
    /// The value representing the thread id.
    pub value: u64,
}

#[cfg(feature = "std")]
static STREAM_COUNT: AtomicU64 = AtomicU64::new(0);

impl StreamId {
    /// Get the current thread id.
    pub fn current() -> Self {
        Self {
            #[cfg(feature = "std")]
            value: Self::from_current_thread(),
            #[cfg(not(feature = "std"))]
            value: 0,
        }
    }

    #[cfg(feature = "std")]
    fn from_current_thread() -> u64 {
        std::thread_local! {
            static ID: std::cell::OnceCell::<u64> = const { std::cell::OnceCell::new() };
        };

        // Getting the current thread is expensive, so we cache the value into a thread local
        // variable, which is very fast.
        ID.with(|cell| {
            *cell.get_or_init(|| STREAM_COUNT.fetch_add(1, core::sync::atomic::Ordering::Acquire))
        })
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}
