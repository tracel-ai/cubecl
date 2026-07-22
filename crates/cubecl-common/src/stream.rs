//! Explicit stream creation, decoupled from threads.
//!
//! A *stream* is a logical unit of execution identified by a [`StreamId`]. Each
//! stream owns a single memory pool, so running work *on* a stream pins that
//! work to the stream's pool regardless of which OS thread runs it. This is
//! what lets memory-heavy workloads reuse one pool instead of implicitly
//! growing a separate pool per thread.
//!
//! The API mirrors the thread API on purpose. [`Stream::spawn`] takes a
//! [`StreamBacking`], a closure, and an optional stream number, and returns a
//! [`StreamHandle`] you can [`join`](StreamHandle::join):
//!
//! ```
//! use cubecl_common::stream::{Stream, StreamBacking};
//!
//! // Run on a fresh stream, synchronously.
//! Stream::spawn(StreamBacking::Sequential, || { /* ... */ }, None).join().unwrap();
//!
//! // Run on the stream with number 5 — the same number always maps to the
//! // same stream (and pool), so it can be shared across threads.
//! Stream::spawn(StreamBacking::Sequential, || { /* ... */ }, 5).join().unwrap();
//! ```

use crate::stream_id::StreamId;

/// The result of [`joining`](StreamHandle::join) a stream.
///
/// Mirrors [`std::thread::Result`]: the error carries the panic payload of a
/// thread-backed stream. Sequential streams never fail.
#[cfg(multi_threading)]
pub type JoinResult<T> = std::thread::Result<T>;

/// The result of [`joining`](StreamHandle::join) a stream.
///
/// On targets without threads a stream can only run sequentially and therefore
/// never fails, so the error is [`Infallible`](core::convert::Infallible).
#[cfg(not(multi_threading))]
pub type JoinResult<T> = core::result::Result<T, core::convert::Infallible>;

/// How a stream's work is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamBacking {
    /// Run the closure synchronously on the calling thread.
    ///
    /// The only backing available on targets without threads (e.g. wasm).
    Sequential,
    /// Run the closure on a dedicated worker thread.
    #[cfg(multi_threading)]
    Thread,
}

impl StreamBacking {
    /// The best backing available on the current target: a worker thread when
    /// threads are available, otherwise sequential execution.
    pub const fn preferred() -> Self {
        #[cfg(multi_threading)]
        {
            StreamBacking::Thread
        }
        #[cfg(not(multi_threading))]
        {
            StreamBacking::Sequential
        }
    }
}

impl Default for StreamBacking {
    fn default() -> Self {
        Self::preferred()
    }
}

/// Namespace for spawning work onto streams. Mirrors [`std::thread`].
///
/// This type is never instantiated; use the associated [`spawn`](Stream::spawn)
/// function.
pub enum Stream {}

impl Stream {
    /// Spawn `f` onto a stream using the given `backing`, returning a
    /// [`StreamHandle`] to its result.
    ///
    /// `stream` selects which stream the work runs on:
    /// - `None` — a fresh, automatically-assigned stream.
    /// - a `u64` number — the stream with that number. The same number always
    ///   refers to the same stream (and memory pool), so spawning with the same
    ///   number from different threads keeps their work on one pool.
    ///
    /// For [`StreamBacking::Sequential`] the closure runs before this returns
    /// and the handle already holds the result. For [`StreamBacking::Thread`]
    /// the closure runs on a fresh worker thread pinned to the stream.
    ///
    /// The `Send`/`'static` bounds match [`std::thread::spawn`] so the same
    /// signature can back either variant.
    pub fn spawn<F, T>(
        backing: StreamBacking,
        f: F,
        stream: impl Into<Option<u64>>,
    ) -> StreamHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let id = match stream.into() {
            Some(number) => StreamId::from_number(number),
            None => StreamId::fresh(),
        };

        match backing {
            StreamBacking::Sequential => StreamHandle {
                inner: StreamHandleInner::Ready(id.executes(f)),
            },
            #[cfg(multi_threading)]
            StreamBacking::Thread => StreamHandle {
                inner: StreamHandleInner::Thread(std::thread::spawn(move || id.executes(f))),
            },
        }
    }
}

/// A handle to work running (or already run) on a stream.
///
/// Returned by [`Stream::spawn`]; call [`join`](Self::join) to wait for the
/// result. Mirrors [`std::thread::JoinHandle`] so both backings share one
/// interface.
pub struct StreamHandle<T> {
    inner: StreamHandleInner<T>,
}

enum StreamHandleInner<T> {
    /// Sequential backing: the result is already computed.
    Ready(T),
    /// Thread backing: waiting on a worker thread.
    #[cfg(multi_threading)]
    Thread(std::thread::JoinHandle<T>),
}

impl<T> StreamHandle<T> {
    /// Wait for the stream's work to finish and return its result.
    ///
    /// Mirrors [`std::thread::JoinHandle::join`]: a thread-backed stream whose
    /// closure panicked returns `Err` with the panic payload. Sequential
    /// streams always return `Ok`.
    pub fn join(self) -> JoinResult<T> {
        match self.inner {
            StreamHandleInner::Ready(value) => Ok(value),
            #[cfg(multi_threading)]
            StreamHandleInner::Thread(handle) => handle.join(),
        }
    }
}

impl<T> core::fmt::Debug for StreamHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamHandle").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_sequential_runs_eagerly_and_joins() {
        let handle = Stream::spawn(StreamBacking::Sequential, || 2 + 2, None);
        assert_eq!(handle.join().unwrap(), 4);
    }

    #[test]
    fn explicit_number_pins_the_task_to_that_stream() {
        let seen = Stream::spawn(StreamBacking::Sequential, StreamId::current, 5)
            .join()
            .unwrap();
        assert_eq!(seen, StreamId::from_number(5));
    }

    #[test]
    fn none_uses_a_fresh_stream_distinct_from_user_numbers() {
        let seen = Stream::spawn(StreamBacking::Sequential, StreamId::current, None)
            .join()
            .unwrap();
        // Fresh ids count up from 0 (top bit clear); user numbers set the top
        // bit, so the two spaces can never alias.
        assert_ne!(seen, StreamId::from_number(0));
    }

    #[cfg(multi_threading)]
    #[test]
    fn spawn_thread_pins_the_worker_to_the_stream() {
        let seen = Stream::spawn(StreamBacking::Thread, StreamId::current, 99)
            .join()
            .unwrap();
        assert_eq!(seen, StreamId::from_number(99));
    }

    #[cfg(multi_threading)]
    #[test]
    fn spawn_thread_join_returns_err_on_panic() {
        let handle = Stream::spawn(StreamBacking::Thread, || panic!("boom"), None);
        assert!(handle.join().is_err());
    }
}
