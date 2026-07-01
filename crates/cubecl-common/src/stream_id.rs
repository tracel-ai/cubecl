/// Unique identifier that can represent a stream.
///
/// Historically a `StreamId` was *thread identity*: [`StreamId::current`] lazily
/// minted a fresh id per OS thread and cached it. That behavior is preserved for
/// back-compat, but a `StreamId` is no longer tied to a thread — [`StreamId::new`]
/// mints an explicit id that can be moved across threads (or used on wasm), and
/// [`StreamId::executes`] scopes it as the *ambient current stream* for the
/// duration of a closure.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct StreamId {
    /// The value representing the stream.
    pub value: u64,
}

impl StreamId {
    /// Allocate a fresh, globally-unique stream id, independent of the current thread.
    ///
    /// Unlike [`StreamId::current`], this never reads or writes the ambient
    /// current-stream cell — it just mints the next value from the global
    /// counter. This is the explicit-stream constructor used by `Stream`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            value: ambient::next_value(),
        }
    }

    /// Executes `f` on this stream, restoring the previous stream afterward.
    ///
    /// The previous [`StreamId`] is saved before the call and restored on
    /// return — including on unwind — so the caller never has to manage
    /// raw `swap` pairs.
    pub fn executes<F, T>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        struct Guard(StreamId);

        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe {
                    StreamId::swap(self.0);
                }
            }
        }

        let old = unsafe { StreamId::swap(self) };
        let guard = Guard(old);

        let returned = f();
        core::mem::drop(guard);
        returned
    }

    /// Get the current ambient stream id.
    ///
    /// If no stream is currently set on this thread, a fresh id is lazily minted
    /// and cached, preserving the historical thread-per-stream behavior for
    /// callers that never opt into explicit streams.
    pub fn current() -> Self {
        let value = match ambient::get() {
            Some(value) => value,
            None => {
                let value = ambient::next_value();
                ambient::set(Some(value));
                value
            }
        };

        Self { value }
    }

    /// Swap the current stream id for the given one, returning the previous one.
    ///
    /// # Safety
    ///
    /// Unknown at this point, don't use that if you don't know what you are doing.
    pub unsafe fn swap(stream: StreamId) -> StreamId {
        let old = Self::current();
        ambient::set(Some(stream.value));
        old
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}

/// Backing store for the global stream counter and the ambient current-stream cell.
///
/// Two implementations keep `swap`/`executes`/`current` working uniformly across
/// every build:
///
/// * `multi_threading` (`std` && non-wasm): a process-global atomic counter plus a
///   `thread_local!` ambient cell, so each thread has its own current stream (the
///   historical behavior).
/// * `not(multi_threading)` (`no_std` or wasm): a single-threaded global cell for
///   both the counter and the ambient value. This is what lets multiple explicit
///   streams coexist on single-threaded / wasm targets (where `swap` used to be a
///   no-op). It is sound only because these targets are assumed single-threaded —
///   the same assumption the rest of the crate already encodes in the
///   `multi_threading` cfg.
#[cfg(multi_threading)]
mod ambient {
    use core::cell::Cell;
    use core::sync::atomic::{AtomicU64, Ordering};

    static STREAM_COUNT: AtomicU64 = AtomicU64::new(0);

    std::thread_local! {
        static CURRENT: Cell<Option<u64>> = const { Cell::new(None) };
    }

    /// Mint the next globally-unique stream value.
    pub(super) fn next_value() -> u64 {
        STREAM_COUNT.fetch_add(1, Ordering::Relaxed)
    }

    /// Read the ambient current-stream value for this thread.
    pub(super) fn get() -> Option<u64> {
        CURRENT.with(|cell| cell.get())
    }

    /// Set the ambient current-stream value for this thread.
    pub(super) fn set(value: Option<u64>) {
        CURRENT.with(|cell| cell.set(value));
    }
}

#[cfg(not(multi_threading))]
mod ambient {
    use core::cell::Cell;

    /// Wrapper that makes a `Cell` usable inside a `static`. Sound because
    /// `not(multi_threading)` builds (`no_std` or wasm) are single-threaded
    /// (see module docs).
    struct SingleThreaded<T>(T);

    // SAFETY: not(multi_threading) builds (no_std or wasm) never access these
    // statics from more than one thread.
    unsafe impl<T> Sync for SingleThreaded<T> {}

    static STREAM_COUNT: SingleThreaded<Cell<u64>> = SingleThreaded(Cell::new(0));
    static CURRENT: SingleThreaded<Cell<Option<u64>>> = SingleThreaded(Cell::new(None));

    /// Mint the next globally-unique stream value.
    pub(super) fn next_value() -> u64 {
        let value = STREAM_COUNT.0.get();
        STREAM_COUNT.0.set(value + 1);
        value
    }

    /// Read the ambient current-stream value.
    pub(super) fn get() -> Option<u64> {
        CURRENT.0.get()
    }

    /// Set the ambient current-stream value.
    pub(super) fn set(value: Option<u64>) {
        CURRENT.0.set(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new()` mints distinct ids straight from the counter, without touching the
    /// ambient current-stream cell.
    #[test]
    fn new_yields_distinct_ids_without_setting_current() {
        let a = StreamId::new();
        let b = StreamId::new();
        assert_ne!(a.value, b.value, "explicit ids must be unique");
    }

    /// `executes` scopes the ambient id for the duration of the closure and
    /// restores it afterward.
    #[test]
    fn executes_swaps_and_restores_current() {
        let outer = StreamId::current();
        let scoped = StreamId::new();
        assert_ne!(outer.value, scoped.value);

        scoped.executes(|| {
            assert_eq!(
                StreamId::current(),
                scoped,
                "current() must observe the scoped id inside executes"
            );
        });

        assert_eq!(
            StreamId::current(),
            outer,
            "current() must be restored after executes"
        );
    }

    /// Nested `executes` calls restore the correct id at each level.
    #[test]
    fn executes_nests_correctly() {
        let a = StreamId::new();
        let b = StreamId::new();

        a.executes(|| {
            assert_eq!(StreamId::current(), a);
            b.executes(|| {
                assert_eq!(StreamId::current(), b);
            });
            assert_eq!(StreamId::current(), a, "inner scope must restore to a");
        });
    }
}
