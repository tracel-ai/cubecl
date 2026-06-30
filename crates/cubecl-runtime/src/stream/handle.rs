use cubecl_common::stream_id::StreamId;

/// A first-class, thread-decoupled stream handle.
///
/// A `Stream` is an explicit [`StreamId`] you run work under. Running work
/// through it scopes the id as the *ambient current stream* (via
/// [`StreamId::executes`]) for the duration of the closure, so every layer that
/// reads the ambient id — the cubecl memory pool of a default
/// [`ComputeClient`](crate::client::ComputeClient) and the `burn-fusion`
/// op-queue alike — observes this one stream. Pinning N threads to a single
/// `Stream` therefore collapses them onto one stream and one memory pool.
///
/// The work runs on the calling thread; many threads holding clones of the same
/// `Stream` can run concurrently and all share its id (hence one pool). Cloning
/// is cheap — the id is `Copy`.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    id: StreamId,
}

impl Stream {
    /// Creates a stream with a fresh, explicit id.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { id: StreamId::new() }
    }

    /// The explicit id backing this stream.
    pub fn id(&self) -> StreamId {
        self.id
    }

    /// Runs `f` on this stream, scoping its id as the ambient current stream for
    /// the duration of the call.
    ///
    /// Inside `f`, both [`StreamId::current`] and a default
    /// [`ComputeClient`](crate::client::ComputeClient) observe this stream's id.
    pub fn run<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.id.executes(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closure run through a stream observes its id as the ambient current stream.
    #[test]
    fn run_observes_stream_id() {
        let stream = Stream::new();
        let observed = stream.run(StreamId::current);
        assert_eq!(observed, stream.id());
    }

    /// Cloned handles share the same id, so work routed through any clone lands on
    /// one stream.
    #[test]
    fn clones_share_one_id() {
        let stream = Stream::new();
        let clone = stream;
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
}
