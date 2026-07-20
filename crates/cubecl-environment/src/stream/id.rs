#[cfg(stream_local)]
use core::cell::Cell;
#[cfg(stream_local)]
use core::sync::atomic::AtomicU64;

#[cfg(stream_local)]
use super::StreamPolicy;

/// Unique identifier representing the stream on which work is submitted.
///
/// How the current stream is resolved depends on the active
/// [`StreamPolicy`](super::StreamPolicy) and on any explicit override installed
/// with [`StreamId::executes`] or [`Stream::enter`](super::Stream::enter).
#[derive(
    Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct StreamId {
    /// The value representing the stream id.
    pub value: u64,
}

#[cfg(stream_local)]
static STREAM_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(stream_local)]
std::thread_local! {
    /// Explicitly scoped stream override, installed by [`StreamId::executes`].
    static OVERRIDE: Cell<Option<u64>> = const { Cell::new(None) };
    /// Lazily assigned per-thread default stream.
    static DEFAULT: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Replaces the current stream override, returning the previous one.
///
/// `None` means "no override": [`StreamId::current`] falls back to the active
/// policy. Keeping the override separate from the per-thread default is what
/// allows a scoped override to be fully undone, even on threads that never had
/// a default assigned.
#[cfg(stream_local)]
pub(crate) fn set_override(value: Option<u64>) -> Option<u64> {
    OVERRIDE.with(|cell| cell.replace(value))
}

impl StreamId {
    /// Executes `f` on this stream, restoring the previous stream afterward.
    ///
    /// The previous state is saved before the call and restored on return —
    /// including on unwind — so the caller never has to manage raw override
    /// pairs. Restoring also works when no stream was active before the call.
    pub fn executes<F, T>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        #[cfg(stream_local)]
        {
            struct Guard(Option<u64>);

            impl Drop for Guard {
                fn drop(&mut self) {
                    set_override(self.0);
                }
            }

            let _guard = Guard(set_override(Some(self.value)));
            f()
        }

        #[cfg(not(stream_local))]
        f()
    }

    /// Get the current stream id.
    ///
    /// Resolution order:
    /// 1. An explicit override installed by [`StreamId::executes`].
    /// 2. The active [`StreamPolicy`](super::StreamPolicy): a stable per-task
    ///    id under [`PerTask`](super::StreamPolicy::PerTask), stream `0` under
    ///    [`Single`](super::StreamPolicy::Single), or a lazily assigned
    ///    per-thread id otherwise.
    pub fn current() -> Self {
        #[cfg(stream_local)]
        {
            if let Some(value) = OVERRIDE.with(|cell| cell.get()) {
                return Self { value };
            }

            match super::policy() {
                StreamPolicy::Single => Self { value: 0 },
                StreamPolicy::PerTask => Self::per_task(),
                StreamPolicy::PerThread => Self::per_thread(),
            }
        }

        #[cfg(not(stream_local))]
        Self { value: 0 }
    }

    /// Allocate a fresh stream id, distinct from every per-thread default.
    ///
    /// On no-std targets there is a single stream, so this returns id `0`.
    pub fn allocate() -> Self {
        #[cfg(stream_local)]
        {
            Self {
                value: STREAM_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
            }
        }

        #[cfg(not(stream_local))]
        Self { value: 0 }
    }

    #[cfg(stream_local)]
    fn per_thread() -> Self {
        DEFAULT.with(|cell| match cell.get() {
            Some(value) => Self { value },
            None => {
                let new = Self::allocate();
                cell.set(Some(new.value));
                new
            }
        })
    }

    #[cfg(all(stream_local, tokio_rt))]
    fn per_task() -> Self {
        match tokio::task::try_id() {
            Some(id) => {
                use core::hash::BuildHasher;

                let hash = foldhash::fast::FixedState::default().hash_one(id);

                // The high bit namespaces task-derived ids away from the
                // counter-based thread and manual ids. A hash collision or a
                // recycled tokio task id only merges two logical streams onto
                // one backend stream, which is safe (FIFO ordering), while a
                // single task always keeps one stable id across thread hops.
                Self {
                    value: hash | (1 << 63),
                }
            }
            // Not inside a tokio task: behave like the per-thread policy.
            None => Self::per_thread(),
        }
    }

    #[cfg(all(stream_local, not(tokio_rt)))]
    fn per_task() -> Self {
        #[cfg(feature = "std")]
        {
            use std::sync::Once;

            static WARN: Once = Once::new();
            WARN.call_once(|| {
                log::warn!(
                    "Stream policy 'per-task' requires the 'tokio' feature of cubecl-environment; falling back to 'per-thread'."
                );
            });
        }

        Self::per_thread()
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}

#[cfg(all(test, stream_local))]
mod tests {
    use super::*;

    #[test]
    fn executes_restores_previous_override() {
        let outer = StreamId { value: 1_000_000 };
        let inner = StreamId { value: 2_000_000 };

        outer.executes(|| {
            assert_eq!(StreamId::current(), outer);
            inner.executes(|| {
                assert_eq!(StreamId::current(), inner);
            });
            assert_eq!(StreamId::current(), outer);
        });
    }

    #[test]
    fn executes_restores_no_override_state() {
        // Regression: restoring after the outermost `executes` must return to
        // "no override", not pin the resolved id onto the thread.
        let scoped = StreamId { value: 500_000 };

        scoped.executes(|| {
            assert_eq!(StreamId::current(), scoped);
        });

        assert_eq!(OVERRIDE.with(|cell| cell.get()), None);
        assert_ne!(StreamId::current(), scoped);
    }

    #[test]
    fn current_is_stable_on_one_thread() {
        let _guard = crate::stream::tests_policy_lock();

        assert_eq!(StreamId::current(), StreamId::current());
    }

    #[test]
    fn allocate_returns_distinct_ids() {
        assert_ne!(StreamId::allocate(), StreamId::allocate());
    }
}
