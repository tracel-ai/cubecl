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

#[cfg(feature = "std")]
std::thread_local! {
        static ID: std::cell::RefCell::<Option<u64>> = const { std::cell::RefCell::new(None) };
}

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
    /// Swap the current stream id for the given one.
    ///
    /// # Safety
    ///
    /// Unknown at this point, don't use that if you don't know what you are doing.
    pub unsafe fn swap(stream: StreamId) -> StreamId {
        let old = Self::current();
        ID.with(|cell| {
            let mut val = cell.borrow_mut();
            *val = Some(stream.value)
        });

        old
    }

    #[cfg(feature = "std")]
    fn from_current_thread() -> u64 {
        ID.with(|cell| {
            let mut val = cell.borrow_mut();
            match val.as_mut() {
                Some(val) => *val,
                None => {
                    let new = STREAM_COUNT.fetch_add(1, core::sync::atomic::Ordering::Acquire);
                    *val = Some(new);
                    new
                }
            }
        })
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}
