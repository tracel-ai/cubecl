use alloc::sync::Arc;
use cubecl_common::stream::{InlineExecutor, StreamExecutor};
use cubecl_common::stream_id::StreamId;

/// A first-class, thread-decoupled stream handle.
///
/// A `Stream` pairs an explicit [`StreamId`] with a pluggable
/// [`StreamExecutor`]. Running work through it scopes the id as the *ambient
/// current stream* (via [`StreamId::executes`]) for the duration of the closure,
/// so every layer that reads the ambient id — the cubecl memory pool of a default
/// [`ComputeClient`](crate::client::ComputeClient) and the `burn-fusion` op-queue
/// alike — observes this one stream. Pinning N threads to a single `Stream`
/// therefore collapses them onto one stream and one memory pool.
///
/// Cloning is cheap: the id is `Copy` and the executor is shared behind an
/// [`Arc`]. All clones drive the same underlying executor.
#[derive(Clone)]
pub struct Stream {
    id: StreamId,
    exec: Arc<dyn StreamExecutor>,
}

impl Stream {
    /// Creates a stream with a fresh id that runs work inline on the calling thread.
    ///
    /// This is the wasm-safe default: several threads holding clones of the same
    /// `Stream` all run inline and share its id (hence one pool).
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_executor(Arc::new(InlineExecutor))
    }

    /// Creates a stream with a fresh id backed by the given executor.
    pub fn with_executor(exec: Arc<dyn StreamExecutor>) -> Self {
        Self {
            id: StreamId::new(),
            exec,
        }
    }

    /// Creates a stream with a fresh id backed by a dedicated worker thread.
    ///
    /// Every closure passed to [`Stream::run`] is serialized onto that worker.
    #[cfg(multi_threading)]
    pub fn thread() -> Self {
        Self::with_executor(Arc::new(cubecl_common::stream::ThreadExecutor::new()))
    }

    /// The explicit id backing this stream.
    pub fn id(&self) -> StreamId {
        self.id
    }

    /// Runs `f` on this stream, scoping its id as the ambient current stream for
    /// the duration of the call, and blocks until it returns.
    ///
    /// Inside `f`, both [`StreamId::current`] and a default
    /// [`ComputeClient`](crate::client::ComputeClient) observe this stream's id.
    pub fn run<'a, R, F>(&self, f: F) -> R
    where
        R: Send + 'static,
        F: FnOnce() -> R + Send + 'a,
    {
        let id = self.id;
        let job: cubecl_common::stream::ErasedJob<'a> = alloc::boxed::Box::new(move || {
            alloc::boxed::Box::new(id.executes(f)) as alloc::boxed::Box<dyn core::any::Any + Send>
        });

        *self
            .exec
            .run_blocking(job)
            .downcast::<R>()
            .expect("the stream executor must return the job's result type")
    }
}

impl core::fmt::Debug for Stream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream").field("id", &self.id).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closure run inline observes the stream's id as the ambient current stream.
    #[test]
    fn inline_run_observes_stream_id() {
        let stream = Stream::new();
        let observed = stream.run(StreamId::current);
        assert_eq!(observed, stream.id());
    }

    /// Cloned handles share the same id, so work routed through any clone lands on
    /// one stream.
    #[test]
    fn clones_share_one_id() {
        let stream = Stream::new();
        let clone = stream.clone();
        assert_eq!(stream.id(), clone.id());
        assert_eq!(clone.run(StreamId::current), stream.id());
    }

    /// `run` restores the ambient id after returning.
    #[test]
    fn run_restores_ambient_id() {
        let outer = StreamId::current();
        let stream = Stream::new();
        assert_ne!(outer, stream.id());
        stream.run(|| {});
        assert_eq!(StreamId::current(), outer);
    }

    /// Same guarantee when the stream is backed by a dedicated worker thread: the
    /// closure runs off the caller thread yet still observes the stream's id.
    #[cfg(multi_threading)]
    #[test]
    fn thread_run_observes_stream_id_on_worker() {
        let stream = Stream::thread();
        let caller = std::thread::current().id();

        let (id, worker) = stream.run(|| (StreamId::current(), std::thread::current().id()));

        assert_eq!(id, stream.id(), "the closure observes the stream id");
        assert_ne!(caller, worker, "thread() stream must run on its worker");
    }
}
