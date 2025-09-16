use crate::{memory_management::SliceId, server::Binding};
use cubecl_common::stream_id::StreamId;
use hashbrown::HashMap;

/// Trait defining the backend operations for managing streams and events.
///
/// This trait provides the necessary methods for initializing streams, flushing them to create events,
/// and waiting on events for synchronization purposes.
pub trait StreamBackend {
    /// The type representing a stream in this backend.
    type Stream: core::fmt::Debug;
    /// The type representing an event in this backend.
    type Event;

    /// Initializes and returns a new stream associated with the given stream ID.
    fn init_stream(stream_id: StreamId) -> Self::Stream;
    /// Flushes the given stream, ensuring all pending operations are submitted, and returns an event
    /// that can be used for synchronization.
    fn flush(stream: &mut Self::Stream) -> Self::Event;
    /// Makes the stream wait for the specified event to complete before proceeding with further operations.
    fn wait_event(stream: &mut Self::Stream, event: Self::Event);
    /// Wait for the given event synching the CPU.
    fn wait_event_sync(event: Self::Event);
}

/// Manages multiple streams with synchronization logic based on shared bindings.
///
/// This struct handles the creation and alignment of streams to ensure proper synchronization
/// when bindings (e.g., buffers) are shared across different streams.
#[derive(Debug)]
pub struct MultiStream<B: StreamBackend> {
    /// The map of stream IDs to their corresponding stream wrappers.
    streams: HashMap<StreamId, StreamWrapper<B>>,
}

/// A wrapper around a backend stream that includes synchronization metadata.
///
/// This includes the stream itself, a map of last synchronized cursors from other streams,
/// and the current cursor position for this stream.
pub struct StreamWrapper<B: StreamBackend> {
    /// The underlying backend stream.
    pub stream: B::Stream,
    /// The current cursor position, representing the logical progress or version of operations on this stream.
    pub cursor: u64,
    /// A map tracking the last synchronized cursor positions from other streams.
    last_synced: HashMap<StreamId, u64>,
    /// Shared bindings that need to be cleaned.
    shareds: Vec<SharedBindings<B>>,
    /// The number of bindings that are shared.
    num_shared: u64,
    /// A map tracking the last synchronized cursor positions from other streams.
    last_gc: u64,
}

struct SharedBindings<B: StreamBackend> {
    /// All the bindings shared in a single execution.
    ///
    /// We keep the binding alive so it won't be allocated in other concurrent streams.
    _batch: Vec<SliceId>,
    /// The event to sync making sure the bindings in the batch are ready to be reused by other streams.
    event: Option<B::Event>,
}

impl<B: StreamBackend> MultiStream<B> {
    /// Creates an empty multi-stream.
    pub fn new() -> Self {
        Self {
            streams: Default::default(),
        }
    }

    /// Get the current stream not linked with any [Binding].
    ///
    /// # Note
    ///
    /// The cursor will still be increased.
    pub fn get(&mut self, stream_id: StreamId) -> &mut StreamWrapper<B> {
        self.resolve(stream_id, [].into_iter())
    }

    /// Resolves and returns a mutable reference to the stream for the given ID, performing any necessary
    /// alignment based on the provided bindings.
    ///
    /// This method ensures that the stream is synchronized with any shared bindings from other streams
    /// before returning the stream reference.
    pub fn resolve<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> &mut StreamWrapper<B> {
        self.align_streams(stream_id, bindings);

        let stream = self.streams.get_mut(&stream_id).expect("Stream to exist");

        stream.cursor += 1;
        stream.maybe_run_gc();

        stream
    }

    /// Aligns the target stream with other streams based on shared bindings.
    ///
    /// This initializes the stream if it doesn't exist, analyzes which originating streams need flushing
    /// for synchronization, flushes them, and waits on the events in the target stream.
    fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) {
        if !self.streams.contains_key(&stream_id) {
            let stream = B::init_stream(stream_id);
            let stream = StreamWrapper {
                stream,
                cursor: 0,
                last_synced: Default::default(),
                shareds: Default::default(),
                num_shared: 0,
                last_gc: 0,
            };
            self.streams.insert(stream_id, stream);
        }

        let analysis = self.update_shared_bindings(stream_id, bindings);

        self.apply_analysis(stream_id, analysis);
    }

    /// Update and analyzes the bindings to determine which streams need alignment (flushing and waiting).
    ///
    /// This checks for shared bindings from other streams and determines if synchronization is needed
    /// based on cursor positions.
    pub(crate) fn update_shared_bindings<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> SharedBindingAnalysis {
        let mut analysis = SharedBindingAnalysis::default();
        let current = self.streams.get_mut(&stream_id).unwrap();
        current.ensure_shared_events();

        for binding in bindings {
            if stream_id != binding.stream {
                if let Some(last_synced) = current.last_synced.get(&binding.stream) {
                    if *last_synced < binding.cursor {
                        analysis.shared(binding);
                    }
                } else {
                    analysis.shared(binding);
                }
            }
        }

        analysis
    }

    pub(crate) fn apply_analysis(&mut self, stream_id: StreamId, analysis: SharedBindingAnalysis) {
        if analysis.slices.is_empty() {
            return;
        }

        log::info!("Analysis for stream {stream_id} => {analysis:?}");

        let mut events = Vec::with_capacity(analysis.slices.len());

        for (origin, slices) in analysis.slices {
            let stream = self.streams.get_mut(&origin).unwrap();
            let event = B::flush(&mut stream.stream);

            stream.num_shared += slices.len() as u64;
            stream.shareds.push(SharedBindings {
                _batch: slices,
                event: None,
            });

            events.push(((origin, stream.cursor), event));
        }

        let stream = self.streams.get_mut(&stream_id).unwrap();
        for ((stream_origin, cursor_origin), event) in events {
            stream.last_synced.insert(stream_origin, cursor_origin);

            // B::wait_event(&mut stream.stream, event);
            B::wait_event_sync(event);
        }
    }
}

const GC_BATCH_SIZE: usize = 2;
const GC_MAX_QUEUED: usize = 32;
const GC_MAX_QUEUED_FACTOR: usize = 2;

impl<B: StreamBackend> StreamWrapper<B> {
    /// Maybe run garbage collector on the current stream.
    fn maybe_run_gc(&mut self) {
        let frequency = self.cursor / (self.num_shared + 1);

        let should_run_time = frequency > self.cursor - self.last_gc;
        let should_run_batch = self.shareds.len() >= GC_MAX_QUEUED;

        if should_run_time || should_run_batch {
            self.last_gc = self.cursor;
            let batch_size = match should_run_batch {
                true => GC_MAX_QUEUED_FACTOR * GC_BATCH_SIZE,
                false => GC_BATCH_SIZE,
            };

            self.run_gc(batch_size);
        }
    }

    fn ensure_shared_events(&mut self) {
        for shared in self.shareds.iter_mut().rev() {
            match shared.event {
                Some(..) => break,
                None => {
                    // We need to create an event _after_ the stream has executed something that
                    // needed the bindings. Otherwise bindings that are written too might not be
                    // correctly synchronized.
                    shared.event = Some(B::flush(&mut self.stream));
                }
            }
        }
    }

    fn run_gc(&mut self, batch_size: usize) {
        let batch_size = usize::min(batch_size, self.shareds.len());
        let dropped = self.shareds.drain(0..batch_size);

        // We wait on the last event recorded.
        if let Some(shared) = dropped.last() {
            B::wait_event_sync(shared.event.expect("The event to be initialized"));
        }
    }
}

impl<B: StreamBackend> core::fmt::Debug for StreamWrapper<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamWrapper")
            .field("stream", &self.stream)
            .field("cursor", &self.cursor)
            .field("last_synced", &self.last_synced)
            .field("shareds", &self.shareds.len())
            .field("num_shared", &self.num_shared)
            .finish()
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct SharedBindingAnalysis {
    slices: HashMap<StreamId, Vec<SliceId>>,
}

impl SharedBindingAnalysis {
    fn shared(&mut self, binding: &Binding) {
        match self.slices.get_mut(&binding.stream) {
            Some(bindings) => bindings.push(binding.memory.id().clone()),
            None => {
                self.slices
                    .insert(binding.stream.clone(), vec![binding.memory.id().clone()]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::SliceHandle, server::Handle};

    #[test]
    fn test_analysis_shared_bindings() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_2);

        let mut ms = MultiStream::<TestBackend>::new();
        ms.get(stream_1);
        ms.get(stream_2);

        let analysis = ms.update_shared_bindings(stream_1, [&binding_1, &binding_2].into_iter());

        let mut expected = SharedBindingAnalysis::default();
        expected.shared(&binding_2);

        assert_eq!(analysis, expected);
    }

    #[test]
    fn test_analysis_shared_bindings_2() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_2);
        let binding_3 = binding(stream_1);

        let mut ms = MultiStream::<TestBackend>::new();
        ms.get(stream_1);
        ms.get(stream_2);

        let analysis =
            ms.update_shared_bindings(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());

        let mut expected = SharedBindingAnalysis::default();
        expected.shared(&binding_2);

        assert_eq!(analysis, expected);
    }

    #[test]
    fn test_analysis_no_shared() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_1);
        let binding_3 = binding(stream_1);

        let mut ms = MultiStream::<TestBackend>::new();
        ms.get(stream_1);
        ms.get(stream_2);

        let analysis =
            ms.update_shared_bindings(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());

        let expected = SharedBindingAnalysis::default();

        assert_eq!(analysis, expected);
    }

    #[test]
    fn test_state() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_2);
        let binding_3 = binding(stream_1);

        let mut ms = MultiStream::<TestBackend>::new();
        ms.get(stream_1);
        ms.get(stream_2);

        ms.resolve(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());

        let stream1 = ms.streams.remove(&stream_1).unwrap();
        assert_eq!(stream1.last_synced.get(&stream_2), Some(&1));
        assert!(stream1.shareds.is_empty());
        assert_eq!(stream1.num_shared, 0);
        assert_eq!(stream1.cursor, 2);
        assert_eq!(stream1.last_gc, 0);

        let stream2 = ms.streams.remove(&stream_2).unwrap();
        assert!(stream2.last_synced.is_empty());
        for shared in stream2.shareds {
            assert_eq!(shared._batch, vec![binding_2.memory.id().clone()]);
        }
        assert_eq!(stream2.num_shared, 1);
        assert_eq!(stream2.cursor, 1);
        assert_eq!(stream2.last_gc, 0);
    }

    fn binding(stream: StreamId) -> Binding {
        Handle::new(SliceHandle::new(), None, None, stream, 0, 10).binding()
    }

    struct TestBackend;

    #[derive(Debug)]
    struct TestStream {}

    #[derive(Debug)]
    struct TestEvent {}

    impl StreamBackend for TestBackend {
        type Stream = TestStream;
        type Event = TestEvent;

        fn init_stream(_stream_id: StreamId) -> Self::Stream {
            TestStream {}
        }

        fn flush(_stream: &mut Self::Stream) -> Self::Event {
            TestEvent {}
        }

        fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}

        fn wait_event_sync(_event: Self::Event) {}
    }
}
