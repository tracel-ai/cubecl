use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::StreamId;

/// A manually managed stream identity.
///
/// A [`Stream`] is a pure identity: it owns no backend resources (backend
/// streams are pool-managed by the runtimes) and is freely copyable. Use it to
/// pin work to a stable stream regardless of which thread or task executes it:
///
/// - [`Stream::enter`] runs synchronous work on the stream.
/// - [`Stream::attach`] binds a future to the stream, surviving executor
///   work-stealing on any async runtime.
/// - [`Stream::spawn`] runs a closure on a fresh OS thread bound to a fresh
///   stream (native std only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    id: StreamId,
}

impl Stream {
    /// Creates a new stream with a freshly allocated identity.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            id: StreamId::allocate(),
        }
    }

    /// Adopts an existing stream id.
    pub const fn from_id(id: StreamId) -> Self {
        Self { id }
    }

    /// The underlying stream id.
    pub fn id(&self) -> StreamId {
        self.id
    }

    /// Runs `f` on this stream, restoring the previous stream afterward,
    /// including on unwind.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        self.id.executes(f)
    }

    /// Binds a future to this stream.
    ///
    /// The returned future re-establishes the stream around every poll, so the
    /// binding survives executor work-stealing on any async runtime, without
    /// requiring any particular executor.
    pub fn attach<F: Future>(&self, fut: F) -> StreamFuture<F> {
        StreamFuture {
            id: self.id,
            inner: fut,
        }
    }
}

/// A future bound to a [`Stream`], created with [`Stream::attach`].
pub struct StreamFuture<F> {
    id: StreamId,
    inner: F,
}

impl<F: Future> Future for StreamFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Manual pin projection: `inner` is structurally pinned, `id` is Copy.
        // Safety: `inner` is never moved out of `self` after being pinned.
        let (id, inner) = unsafe {
            let this = self.get_unchecked_mut();
            (this.id, Pin::new_unchecked(&mut this.inner))
        };
        id.executes(|| inner.poll(cx))
    }
}

#[cfg(multi_threading)]
impl Stream {
    /// Runs `f` on a fresh OS thread bound to a fresh stream.
    ///
    /// Everything submitted inside `f` targets the new stream, concurrent with
    /// work on other streams.
    pub fn spawn<T, F>(f: F) -> StreamJoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let stream = Self::new();
        let id = stream.id;
        let handle = std::thread::spawn(move || id.executes(f));

        StreamJoinHandle { stream, handle }
    }
}

/// Handle to a thread spawned with [`Stream::spawn`].
#[cfg(multi_threading)]
#[derive(Debug)]
pub struct StreamJoinHandle<T> {
    stream: Stream,
    handle: std::thread::JoinHandle<T>,
}

#[cfg(multi_threading)]
impl<T> StreamJoinHandle<T> {
    /// The stream the spawned closure runs on.
    pub fn stream(&self) -> Stream {
        self.stream
    }

    /// Waits for the spawned closure to finish, returning its result.
    pub fn join(self) -> std::thread::Result<T> {
        self.handle.join()
    }
}

#[cfg(tokio_rt)]
impl Stream {
    /// Spawns a future on the tokio runtime, bound to a fresh stream.
    ///
    /// Equivalent to `tokio::spawn(stream.attach(fut))`, returning the stream
    /// so callers can relate results to it.
    pub fn spawn_task<F>(fut: F) -> (Stream, tokio::task::JoinHandle<F::Output>)
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let stream = Self::new();
        (stream, tokio::spawn(stream.attach(fut)))
    }
}

/// Spawns a detached future bound to a fresh stream.
///
/// Uses a thread on native, the browser runtime on wasm; panics on no-std.
pub fn spawn_detached(fut: impl Future<Output = ()> + Send + 'static) -> Stream {
    let stream = Stream::new();
    crate::future::spawn_detached(stream.attach(fut));
    stream
}

#[cfg(all(test, multi_threading))]
mod tests {
    use super::*;

    #[test]
    fn enter_pins_the_stream() {
        let stream = Stream::new();
        let current = stream.enter(StreamId::current);
        assert_eq!(current, stream.id());
    }

    #[test]
    fn spawn_runs_on_its_own_stream() {
        let handle = Stream::spawn(StreamId::current);
        let expected = handle.stream().id();
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[cfg(all(test, tokio_rt))]
mod tests_tokio {
    use super::*;
    use crate::stream::StreamPolicy;
    use alloc::vec::Vec;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn attach_keeps_stream_across_awaits() {
        let stream = Stream::new();
        let id = stream.id();

        let checks = stream.attach(async move {
            for _ in 0..32 {
                assert_eq!(StreamId::current(), id);
                tokio::task::yield_now().await;
            }
        });

        tokio::spawn(checks).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    // The guard serializes this test against the other policy-mutating ones, so
    // it has to span the awaits: dropping it earlier is exactly the race it
    // exists to prevent. Nothing awaited here takes that lock, so it can't
    // deadlock.
    #[allow(clippy::await_holding_lock)]
    async fn per_task_ids_are_stable_and_distinct() {
        let _guard = crate::stream::tests_policy_lock();

        crate::stream::set_policy(StreamPolicy::PerTask);

        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(tokio::spawn(async {
                let first = StreamId::current();
                for _ in 0..32 {
                    tokio::task::yield_now().await;
                    assert_eq!(StreamId::current(), first);
                }
                first
            }));
        }

        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.await.unwrap());
        }
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 8, "each task should get its own stream id");

        crate::stream::tests_reset_policy();
    }
}
