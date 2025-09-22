#[cfg(multi_threading)]
use core::sync::atomic::AtomicU64;

/// Unique identifier that can represent a stream based on the current thread id.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct StreamId {
    /// The value representing the thread id.
    pub value: u64,
}

#[cfg(multi_threading)]
static STREAM_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(multi_threading)]
std::thread_local! {
        static ID: std::cell::RefCell::<Option<u64>> = const { std::cell::RefCell::new(None) };
}

impl StreamId {
    /// Get the current thread id.
    pub fn current() -> Self {
        Self {
            #[cfg(multi_threading)]
            value: Self::from_current_thread(),
            #[cfg(not(multi_threading))]
            value: 0,
        }
    }

    /// Swap the current stream id for the given one.
    ///
    /// # Safety
    ///
    /// Unknown at this point, don't use that if you don't know what you are doing.
    pub unsafe fn swap(stream: StreamId) -> StreamId {
        unsafe {
            #[cfg(multi_threading)]
            return Self::swap_multi_thread(stream);

            #[cfg(not(multi_threading))]
            return Self::swap_single_thread(stream);
        }
    }

    #[cfg(multi_threading)]
    unsafe fn swap_multi_thread(stream: StreamId) -> StreamId {
        let old = Self::current();
        ID.with(|cell| {
            let mut val = cell.borrow_mut();
            *val = Some(stream.value)
        });

        old
    }

    #[cfg(not(multi_threading))]
    unsafe fn swap_single_thread(stream: StreamId) -> StreamId {
        stream
    }

    #[cfg(multi_threading)]
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
