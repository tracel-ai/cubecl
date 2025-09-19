use std::sync::mpsc::SyncSender;

use crate::{memory_management::SliceId, server::Binding};
use cubecl_common::stream_id::StreamId;
use hashbrown::HashMap;

/// Trait defining the backend operations for managing streams and events.
///
/// This trait provides the necessary methods for initializing streams, flushing them to create events,
/// and waiting on events for synchronization purposes.
pub trait StreamBackend: 'static {
    /// The type representing a stream in this backend.
    type Stream: core::fmt::Debug;
    /// The type representing an event in this backend.
    type Event: Send + 'static;

    /// Initializes and returns a new stream associated with the given stream ID.
    fn create_stream(&self) -> Self::Stream;
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
    streams: StreamPool<B>,
    max_streams: usize,
    gc: GcThread<B>,
}

/// A wrapper around a backend stream that includes synchronization metadata.
///
/// This includes the stream itself, a map of last synchronized cursors from other streams,
/// and the current cursor position for this stream.
struct StreamWrapper<B: StreamBackend> {
    /// The underlying backend stream.
    stream: B::Stream,
    /// The current cursor position, representing the logical progress or version of operations on this stream.
    cursor: u64,
    /// A map tracking the last synchronized cursor positions from other streams.
    last_synced: HashMap<usize, u64>,
}

/// Streams that are synchronized correctly after a [MultiStream::resolve] is called.
pub struct ResolvedStreams<'a, B: StreamBackend> {
    /// The cursor on the current stream.
    ///
    /// This cursor should be use for new allocations happening on the current stream.
    pub cursor: u64,
    streams: &'a mut StreamPool<B>,
    analysis: SharedBindingAnalysis,
    gc: &'a GcThread<B>,
    /// The current stream where new tasks can be sent safely.
    pub current: StreamId,
}

#[derive(Debug)]
struct GcTask<B: StreamBackend> {
    /// All the bindings shared in a single execution.
    ///
    /// We keep the binding alive so it won't be allocated in other concurrent streams.
    ids: Vec<SliceId>,
    /// The event to sync making sure the bindings in the batch are ready to be reused by other streams.
    event: B::Event,
}

#[derive(Debug)]
struct StreamPool<B: StreamBackend> {
    streams: Vec<Option<StreamWrapper<B>>>,
    backend: B,
    max_streams: usize,
}

impl<B: StreamBackend> StreamPool<B> {
    fn new(backend: B, max_streams: u8) -> Self {
        let mut streams = Vec::with_capacity(max_streams as usize);
        for _ in 0..max_streams + 1 {
            streams.push(None);
        }

        Self {
            streams,
            backend,
            max_streams: max_streams as usize,
        }
    }

    fn get_mut(&mut self, stream_id: &StreamId) -> &mut StreamWrapper<B> {
        let index = self.stream_index(stream_id);

        // The pool is init, can't go over the index because of stream_index.
        unsafe { self.get_mut_index(index) }
    }

    unsafe fn get_mut_index(&mut self, index: usize) -> &mut StreamWrapper<B> {
        unsafe {
            let entry = self.streams.get_unchecked_mut(index);
            match entry {
                Some(val) => val,
                None => {
                    let stream = self.backend.create_stream();
                    let stream = StreamWrapper {
                        stream,
                        cursor: 0,
                        last_synced: Default::default(),
                    };

                    *entry = Some(stream);

                    match entry {
                        Some(val) => val,
                        None => unreachable!(),
                    }
                }
            }
        }
    }

    fn stream_index(&self, stream_id: &StreamId) -> usize {
        stream_index(stream_id, self.max_streams)
    }
    fn get_gc(&mut self) -> &mut B::Stream {
        unsafe { &mut self.get_mut_index(self.max_streams).stream }
    }
}

#[derive(Debug)]
struct GcThread<B: StreamBackend> {
    sender: SyncSender<GcTask<B>>,
}

impl<B: StreamBackend> GcThread<B> {
    fn new() -> GcThread<B> {
        let (sender, recv) = std::sync::mpsc::sync_channel::<GcTask<B>>(32);

        std::thread::spawn(move || {
            while let Ok(event) = recv.recv() {
                B::wait_event_sync(event.event);
                log::info!("Release memory: {:?}", event.ids);
                core::mem::drop(event.ids);
            }
        });

        GcThread { sender }
    }
    fn register(&self, task: GcTask<B>) {
        self.sender.send(task).unwrap()
    }
}

fn stream_index(stream_id: &StreamId, max_streams: usize) -> usize {
    stream_id.value as usize % max_streams
}

impl<'a, B: StreamBackend> ResolvedStreams<'a, B> {
    /// Get the stream associated to the given [stream_id](StreamId).
    pub fn get(&mut self, stream_id: &StreamId) -> &mut B::Stream {
        let stream = self.streams.get_mut(&stream_id);
        &mut stream.stream
    }

    /// Get the stream associated to the [current stream_id](StreamId).
    pub fn current(&mut self) -> &mut B::Stream {
        let stream = self.streams.get_mut(&self.current);
        &mut stream.stream
    }
}

impl<'a, B: StreamBackend> Drop for ResolvedStreams<'a, B> {
    fn drop(&mut self) {
        if self.analysis.slices.is_empty() {
            return;
        }

        for (_index, ids) in self.analysis.slices.drain() {
            let stream = self.streams.get_mut(&self.current);
            let event_orgin = B::flush(&mut stream.stream);

            let stream_gc = self.streams.get_gc();
            B::wait_event(stream_gc, event_orgin);
            let event_gc = B::flush(stream_gc);
            let task = GcTask {
                ids,
                event: event_gc,
            };
            self.gc.register(task);
        }
    }
}

impl<B: StreamBackend> MultiStream<B> {
    /// Creates an empty multi-stream.
    pub fn new(backend: B, max_streams: u8) -> Self {
        Self {
            streams: StreamPool::new(backend, max_streams),
            max_streams: max_streams as usize,
            gc: GcThread::new(),
        }
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
    ) -> ResolvedStreams<'_, B> {
        let analysis = self.align_streams(stream_id, bindings);

        let stream = self.streams.get_mut(&stream_id);
        stream.cursor += 1;

        ResolvedStreams {
            cursor: stream.cursor,
            streams: &mut self.streams,
            current: stream_id,
            analysis,
            gc: &self.gc,
        }
    }

    /// Aligns the target stream with other streams based on shared bindings.
    ///
    /// This initializes the stream if it doesn't exist, analyzes which originating streams need flushing
    /// for synchronization, flushes them, and waits on the events in the target stream.
    fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> SharedBindingAnalysis {
        let analysis = self.update_shared_bindings(stream_id, bindings);

        self.apply_analysis(stream_id, analysis)
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
        // let current = self.streams.get_mut(&stream_id);

        for binding in bindings {
            if stream_id != binding.stream {
                let index = stream_index(&binding.stream, self.max_streams);
                analysis.shared(binding, index);

                // if let Some(last_synced) = current.last_synced.get(&index) {
                //     if *last_synced < binding.cursor {
                //         analysis.shared(binding, index);
                //     }
                // } else {
                //     analysis.shared(binding, index);
                // }
            }
        }

        analysis
    }

    pub(crate) fn apply_analysis(
        &mut self,
        stream_id: StreamId,
        analysis: SharedBindingAnalysis,
    ) -> SharedBindingAnalysis {
        if analysis.slices.is_empty() {
            return analysis;
        }

        log::info!("Analysis for stream {stream_id} => {analysis:?}");
        // println!("Analysis for stream {stream_id} => {analysis:?}");

        let mut events = Vec::with_capacity(analysis.slices.len());

        unsafe {
            for origin in analysis.slices.keys() {
                let stream = self.streams.get_mut_index(*origin);
                let event = B::flush(&mut stream.stream);

                events.push(((origin, stream.cursor), event));
            }
        }

        let stream = self.streams.get_mut(&stream_id);

        for ((stream_origin, cursor_origin), event) in events {
            stream.last_synced.insert(*stream_origin, cursor_origin);

            log::info!("waiting.. {stream_origin}");
            B::wait_event(&mut stream.stream, event);
        }

        analysis
    }
}

impl<B: StreamBackend> core::fmt::Debug for StreamWrapper<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamWrapper")
            .field("stream", &self.stream)
            .field("cursor", &self.cursor)
            .field("last_synced", &self.last_synced)
            .finish()
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct SharedBindingAnalysis {
    slices: HashMap<usize, Vec<SliceId>>,
}

impl SharedBindingAnalysis {
    fn shared(&mut self, binding: &Binding, index: usize) {
        match self.slices.get_mut(&index) {
            Some(bindings) => bindings.push(binding.memory.id().clone()),
            None => {
                self.slices.insert(index, vec![binding.memory.id().clone()]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::SliceHandle, server::Handle};

    const MAX_STREAMS: u8 = 4;

    #[test]
    fn test_analysis_shared_bindings_1() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_2);

        let mut ms = MultiStream::new(TestBackend, MAX_STREAMS);
        ms.resolve(stream_1, [].into_iter());
        ms.resolve(stream_2, [].into_iter());

        let analysis = ms.update_shared_bindings(stream_1, [&binding_1, &binding_2].into_iter());

        let mut expected = SharedBindingAnalysis::default();
        expected.shared(&binding_2, ms.streams.stream_index(&binding_2.stream));

        assert_eq!(analysis, expected);
    }

    #[test]
    fn test_analysis_shared_bindings_2() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_2);
        let binding_3 = binding(stream_1);

        let mut ms = MultiStream::new(TestBackend, 4);
        ms.resolve(stream_1, [].into_iter());
        ms.resolve(stream_2, [].into_iter());

        let analysis =
            ms.update_shared_bindings(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());

        let mut expected = SharedBindingAnalysis::default();
        expected.shared(&binding_2, ms.streams.stream_index(&binding_2.stream));

        assert_eq!(analysis, expected);
    }

    #[test]
    fn test_analysis_no_shared() {
        let stream_1 = StreamId { value: 1 };
        let stream_2 = StreamId { value: 2 };

        let binding_1 = binding(stream_1);
        let binding_2 = binding(stream_1);
        let binding_3 = binding(stream_1);

        let mut ms = MultiStream::new(TestBackend, MAX_STREAMS);
        ms.resolve(stream_1, [].into_iter());
        ms.resolve(stream_2, [].into_iter());

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

        let mut ms = MultiStream::new(TestBackend, MAX_STREAMS);
        ms.resolve(stream_1, [].into_iter());
        ms.resolve(stream_2, [].into_iter());

        ms.resolve(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());

        let stream1 = ms.streams.get_mut(&stream_1);
        let index_2 = stream_index(&stream_2, MAX_STREAMS as usize);
        assert_eq!(stream1.last_synced.get(&index_2), Some(&1));
        assert_eq!(stream1.cursor, 2);

        let stream2 = ms.streams.get_mut(&stream_2);
        assert!(stream2.last_synced.is_empty());
        assert_eq!(stream2.cursor, 1);
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

        fn create_stream(&self) -> Self::Stream {
            TestStream {}
        }

        fn flush(_stream: &mut Self::Stream) -> Self::Event {
            TestEvent {}
        }

        fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}

        fn wait_event_sync(_event: Self::Event) {}
    }
}
