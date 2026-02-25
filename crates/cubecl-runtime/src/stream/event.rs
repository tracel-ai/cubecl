use crate::{
    config::streaming::StreamingLogLevel,
    logging::ServerLogger,
    server::{HandleBinding, HandleId, ServerError},
    stream::{StreamFactory, StreamPool},
};
use core::any::Any;
use cubecl_common::{backtrace::BackTrace, stream_id::StreamId};
use hashbrown::HashMap;
use std::{
    boxed::Box,
    format,
    sync::{Arc, mpsc::SyncSender},
    vec,
    vec::Vec,
};

/// Trait defining the backend operations for managing streams and events.
///
/// This trait provides the necessary methods for initializing streams, flushing them to create events,
/// and waiting on events for synchronization purposes.
pub trait EventStreamBackend: 'static {
    /// The type representing a stream in this backend.
    type Stream: core::fmt::Debug;
    /// The type representing an event in this backend.
    type Event: Send + 'static;

    /// Initializes and returns a new stream associated with the given stream ID.
    fn create_stream(&self) -> Self::Stream;
    /// Returns the cursor of the given handle on the given stream.
    fn handle_cursor(stream: &Self::Stream, handle: &HandleBinding) -> u64;
    /// Returns wheter the stream can access new tasks.
    fn is_healty(stream: &Self::Stream) -> bool;

    /// Flushes the given stream, ensuring all pending operations are submitted, and returns an event
    /// that can be used for synchronization.
    fn flush(stream: &mut Self::Stream) -> Self::Event;
    /// Makes the stream wait for the specified event to complete before proceeding with further operations.
    fn wait_event(stream: &mut Self::Stream, event: Self::Event);
    /// Wait for the given event synching the CPU.
    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError>;
}

/// Manages multiple streams with synchronization logic based on shared bindings.
///
/// This struct handles the creation and alignment of streams to ensure proper synchronization
/// when bindings (e.g., buffers) are shared across different streams.
#[derive(Debug)]
pub struct MultiStream<B: EventStreamBackend> {
    /// The map of stream IDs to their corresponding stream wrappers.
    streams: StreamPool<EventStreamBackendWrapper<B>>,
    /// The logger used by the server.
    pub logger: Arc<ServerLogger>,
    max_streams: usize,
    gc: GcThread<B>,
    shared_bindings_pool: Vec<(HandleId, StreamId, u64)>,
}

/// A wrapper around a backend stream that includes synchronization metadata.
///
/// This includes the stream itself, a map of last synchronized cursors from other streams,
/// and the current cursor position for this stream.
pub(crate) struct StreamWrapper<B: EventStreamBackend> {
    /// The underlying backend stream.
    stream: B::Stream,
    /// The current cursor position, representing the logical progress or version of operations on this stream.
    cursor: u64,
    /// A map tracking the last synchronized cursor positions from other streams.
    last_synced: HashMap<usize, u64>,
}

/// Streams that are synchronized correctly after a [`MultiStream::resolve`] is called.
pub struct ResolvedStreams<'a, B: EventStreamBackend> {
    /// The cursor on the current stream.
    ///
    /// This cursor should be use for new allocations happening on the current stream.
    pub cursor: u64,
    streams: &'a mut StreamPool<EventStreamBackendWrapper<B>>,
    analysis: SharedBindingAnalysis,
    gc: &'a GcThread<B>,
    /// The current stream where new tasks can be sent safely.
    pub current: StreamId,
}

#[derive(Debug)]
/// A task to be enqueue on the gc stream that will be clearned after an event is reached.
pub struct GcTask<B: EventStreamBackend> {
    to_drop: Box<dyn Any + Send + 'static>,
    /// The event to sync making sure the bindings in the batch are ready to be reused by other streams.
    event: B::Event,
}

impl<B: EventStreamBackend> GcTask<B> {
    /// Creates a new task that will be clearned when the event is reached.
    pub fn new<T: Send + 'static>(to_drop: T, event: B::Event) -> Self {
        Self {
            to_drop: Box::new(to_drop),
            event,
        }
    }
}

#[derive(Debug)]
struct EventStreamBackendWrapper<B: EventStreamBackend> {
    backend: B,
}

impl<B: EventStreamBackend> StreamFactory for EventStreamBackendWrapper<B> {
    type Stream = StreamWrapper<B>;

    fn create(&mut self) -> Self::Stream {
        StreamWrapper {
            stream: self.backend.create_stream(),
            cursor: 0,
            last_synced: Default::default(),
        }
    }
}

#[derive(Debug)]
struct GcThread<B: EventStreamBackend> {
    sender: SyncSender<GcTask<B>>,
}

impl<B: EventStreamBackend> GcThread<B> {
    fn new() -> GcThread<B> {
        let (sender, recv) = std::sync::mpsc::sync_channel::<GcTask<B>>(8);

        std::thread::spawn(move || {
            while let Ok(event) = recv.recv() {
                B::wait_event_sync(event.event).unwrap();
                core::mem::drop(event.to_drop);
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

impl<'a, B: EventStreamBackend> ResolvedStreams<'a, B> {
    /// Get the stream associated to the given [`stream_id`](StreamId).
    pub fn get(&mut self, stream_id: &StreamId) -> &mut B::Stream {
        let stream = self.streams.get_mut(stream_id);
        &mut stream.stream
    }

    /// Get the stream associated to the [current `stream_id`](StreamId).
    pub fn current(&mut self) -> &mut B::Stream {
        let stream = self.streams.get_mut(&self.current);
        &mut stream.stream
    }

    /// Enqueue a task to be cleaned.
    pub fn gc(&mut self, gc: GcTask<B>) {
        self.gc.sender.send(gc).unwrap();
    }
}

impl<'a, B: EventStreamBackend> Drop for ResolvedStreams<'a, B> {
    fn drop(&mut self) {
        if self.analysis.slices.is_empty() {
            return;
        }

        let stream = self.streams.get_mut(&self.current);
        let event_origin = B::flush(&mut stream.stream);

        let stream_gc = &mut unsafe { self.streams.get_special(0) }.stream;
        B::wait_event(stream_gc, event_origin);
        let event = B::flush(stream_gc);

        let mut ids = Vec::new();
        self.analysis
            .slices
            .drain()
            .for_each(|item| ids.extend(item.1));

        self.gc.register(GcTask::new(ids, event));
    }
}

impl<B: EventStreamBackend> MultiStream<B> {
    /// Creates an empty multi-stream.
    pub fn new(logger: Arc<ServerLogger>, backend: B, max_streams: u8) -> Self {
        let wrapper = EventStreamBackendWrapper { backend };
        Self {
            streams: StreamPool::new(wrapper, max_streams, 1),
            logger,
            max_streams: max_streams as usize,
            gc: GcThread::new(),
            shared_bindings_pool: Vec::new(),
        }
    }

    /// Enqueue a task to be cleaned.
    pub fn gc(&mut self, gc: GcTask<B>) {
        self.gc.sender.send(gc).unwrap();
    }

    /// Resolves and returns a mutable reference to the stream for the given ID, performing any necessary
    /// alignment based on the provided bindings.
    ///
    /// This method ensures that the stream is synchronized with any shared bindings from other streams
    /// before returning the stream reference.
    pub fn resolve<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a HandleBinding>,
        enfore_healty: bool,
    ) -> Result<ResolvedStreams<'_, B>, ServerError> {
        let analysis = self.align_streams(stream_id, handles);

        let stream = self.streams.get_mut(&stream_id);
        stream.cursor += 1;

        if enfore_healty && !B::is_healty(&stream.stream) {
            return Err(ServerError::ServerUnHealty {
                reason: format!(
                    "Can't resolve the cuda stream since it is currently in an error state"
                ),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(ResolvedStreams {
            cursor: stream.cursor,
            streams: &mut self.streams,
            current: stream_id,
            analysis,
            gc: &self.gc,
        })
    }

    /// Aligns the target stream with other streams based on shared bindings.
    ///
    /// This initializes the stream if it doesn't exist, analyzes which originating streams need flushing
    /// for synchronization, flushes them, and waits on the events in the target stream.
    fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a HandleBinding>,
    ) -> SharedBindingAnalysis {
        let analysis = self.update_shared_bindings(stream_id, handles);

        self.apply_analysis(stream_id, analysis)
    }

    /// Update and analyzes the bindings to determine which streams need alignment (flushing and waiting).
    ///
    /// This checks for shared bindings from other streams and determines if synchronization is needed
    /// based on cursor positions.
    pub(crate) fn update_shared_bindings<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a HandleBinding>,
    ) -> SharedBindingAnalysis {
        // We reset the memory pool for the info.
        self.shared_bindings_pool.clear();

        for handle in handles {
            let index = stream_index(&handle.stream, self.max_streams);
            let stream = unsafe { self.streams.get_mut_index(index) };
            let cursor_handle = B::handle_cursor(&stream.stream, handle);

            // We only add the info to be consider if the handle stream is different from the current
            // stream.
            if handle.stream != stream_id {
                self.shared_bindings_pool
                    .push((handle.id.clone(), handle.stream, cursor_handle));
            }
        }

        let mut analysis = SharedBindingAnalysis::default();
        let current = self.streams.get_mut(&stream_id);

        for (handle_id, stream, cursor) in self.shared_bindings_pool.iter() {
            let index = stream_index(stream, self.max_streams);

            if let Some(last_synced) = current.last_synced.get(&index) {
                if last_synced < cursor {
                    self.logger.log_streaming(
                        |level| matches!(level, StreamingLogLevel::Full),
                        || {
                            format!(
                                "Binding on {} is shared on {} since it's not sync {} < {}",
                                stream, stream_id, last_synced, cursor
                            )
                        },
                    );
                    analysis.shared(handle_id, index);
                }
            } else {
                self.logger.log_streaming(
                    |level| matches!(level, StreamingLogLevel::Full),
                    || {
                        format!(
                            "Binding on {} is shared on {} since it was never synced.",
                            stream, stream_id,
                        )
                    },
                );
                analysis.shared(handle_id, index);
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

            self.logger.log_streaming(
                |level| !matches!(level, StreamingLogLevel::Disabled),
                || format!("Waiting on {stream_origin} from {stream_id}",),
            );

            B::wait_event(&mut stream.stream, event);
        }

        analysis
    }
}

impl<B: EventStreamBackend> core::fmt::Debug for StreamWrapper<B> {
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
    slices: HashMap<usize, Vec<HandleId>>,
}

impl SharedBindingAnalysis {
    fn shared(&mut self, id: &HandleId, index: usize) {
        match self.slices.get_mut(&index) {
            Some(bindings) => bindings.push(id.clone()),
            None => {
                self.slices.insert(index, vec![id.clone()]);
            }
        }
    }
}

// TODO: Fix the tests
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::server::{Handle, HandleId};
//
//     const MAX_STREAMS: u8 = 4;
//
//     #[test_log::test]
//     fn test_analysis_shared_bindings_1() {
//         let logger = Arc::new(ServerLogger::default());
//         let stream_1 = StreamId { value: 1 };
//         let stream_2 = StreamId { value: 2 };
//
//         let binding_1 = binding(stream_1);
//         let binding_2 = binding(stream_2);
//
//         let mut ms = MultiStream::new(logger, TestBackend, MAX_STREAMS);
//         ms.resolve(stream_1, [].into_iter());
//         ms.resolve(stream_2, [].into_iter());
//
//         let analysis = ms.update_shared_bindings(stream_1, [&binding_1, &binding_2].into_iter());
//
//         let mut expected = SharedBindingAnalysis::default();
//         expected.shared(&binding_2, ms.streams.stream_index(&binding_2.stream));
//
//         assert_eq!(analysis, expected);
//     }
//
//     #[test_log::test]
//     fn test_analysis_shared_bindings_2() {
//         let logger = Arc::new(ServerLogger::default());
//         let stream_1 = StreamId { value: 1 };
//         let stream_2 = StreamId { value: 2 };
//
//         let binding_1 = binding(stream_1);
//         let binding_2 = binding(stream_2);
//         let binding_3 = binding(stream_1);
//
//         let mut ms = MultiStream::new(logger, TestBackend, 4);
//         ms.resolve(stream_1, [].into_iter());
//         ms.resolve(stream_2, [].into_iter());
//
//         let analysis =
//             ms.update_shared_bindings(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());
//
//         let mut expected = SharedBindingAnalysis::default();
//         expected.shared(&binding_2, ms.streams.stream_index(&binding_2.stream));
//
//         assert_eq!(analysis, expected);
//     }
//
//     #[test_log::test]
//     fn test_analysis_no_shared() {
//         let logger = Arc::new(ServerLogger::default());
//         let stream_1 = StreamId { value: 1 };
//         let stream_2 = StreamId { value: 2 };
//
//         let binding_1 = binding(stream_1);
//         let binding_2 = binding(stream_1);
//         let binding_3 = binding(stream_1);
//
//         let mut ms = MultiStream::new(logger, TestBackend, MAX_STREAMS);
//         ms.resolve(stream_1, [].into_iter());
//         ms.resolve(stream_2, [].into_iter());
//
//         let analysis =
//             ms.update_shared_bindings(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());
//
//         let expected = SharedBindingAnalysis::default();
//
//         assert_eq!(analysis, expected);
//     }
//
//     #[test_log::test]
//     fn test_state() {
//         let logger = Arc::new(ServerLogger::default());
//         let stream_1 = StreamId { value: 1 };
//         let stream_2 = StreamId { value: 2 };
//
//         let binding_1 = binding(stream_1);
//         let binding_2 = binding(stream_2);
//         let binding_3 = binding(stream_1);
//
//         let mut ms = MultiStream::new(logger, TestBackend, MAX_STREAMS);
//         ms.resolve(stream_1, [].into_iter());
//         ms.resolve(stream_2, [].into_iter());
//
//         ms.resolve(stream_1, [&binding_1, &binding_2, &binding_3].into_iter());
//
//         let stream1 = ms.streams.get_mut(&stream_1);
//         let index_2 = stream_index(&stream_2, MAX_STREAMS as usize);
//         assert_eq!(stream1.last_synced.get(&index_2), Some(&1));
//         assert_eq!(stream1.cursor, 2);
//
//         let stream2 = ms.streams.get_mut(&stream_2);
//         assert!(stream2.last_synced.is_empty());
//         assert_eq!(stream2.cursor, 1);
//     }
//
//     fn binding(stream: StreamId) -> Handle {
//         Handle::new(HandleId::new(), None, None, stream, 10)
//     }
//
//     struct TestBackend;
//
//     #[derive(Debug)]
//     struct TestStream {}
//
//     #[derive(Debug)]
//     struct TestEvent {}
//
//     impl EventStreamBackend for TestBackend {
//         type Stream = TestStream;
//         type Event = TestEvent;
//
//         fn create_stream(&self) -> Self::Stream {
//             TestStream {}
//         }
//
//         fn flush(_stream: &mut Self::Stream) -> Self::Event {
//             TestEvent {}
//         }
//
//         fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}
//
//         fn wait_event_sync(_event: Self::Event) -> Result<(), ExecutionError> {
//             Ok(())
//         }
//     }
// }
