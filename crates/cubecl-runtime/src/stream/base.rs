use crate::server::{Binding, Bindings};
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
#[derive(Debug)]
pub struct StreamWrapper<B: StreamBackend> {
    /// The underlying backend stream.
    pub stream: B::Stream,
    /// The current cursor position, representing the logical progress or version of operations on this stream.
    pub cursor: u64,
    /// A map tracking the last synchronized cursor positions from other streams.
    last_synced: HashMap<StreamId, u64>,
}

impl<B: StreamBackend> MultiStream<B> {
    /// Creates an empty multi-stream.
    pub fn new() -> Self {
        Self {
            streams: Default::default(),
        }
    }

    pub fn get(&mut self, stream_id: StreamId) -> &mut StreamWrapper<B> {
        if !self.streams.contains_key(&stream_id) {
            let stream = B::init_stream(stream_id);
            log::info!("Initialize a new stream {stream:?}");
            let stream = StreamWrapper {
                stream,
                cursor: 0,
                last_synced: Default::default(),
            };
            self.streams.insert(stream_id, stream);
        }
        self.streams.get_mut(&stream_id).expect("Stream to exist")
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
            };
            self.streams.insert(stream_id, stream);
        }

        let analyses = self.analyse_shared_bindings(stream_id, bindings);
        if analyses.is_empty() {
            return;
        }
        log::info!("Analyzes {analyses:?}");

        let mut events = Vec::with_capacity(analyses.len());

        for stream_origin in analyses {
            let stream = self.streams.get_mut(&stream_origin).unwrap();
            let event = B::flush(&mut stream.stream);

            events.push(((stream_origin, stream.cursor), event));
        }

        let stream = self.streams.get_mut(&stream_id).unwrap();

        for ((stream_origin, cursor_origin), event) in events {
            log::info!(
                "Align stream {:?}[{:?}] with {stream_origin:?}[{cursor_origin:?}]",
                stream.stream,
                stream.cursor
            );
            stream.last_synced.insert(stream_origin, cursor_origin);

            B::wait_event(&mut stream.stream, event);
        }
    }

    /// Analyzes the bindings to determine which streams need alignment (flushing and waiting).
    ///
    /// This checks for shared bindings from other streams and determines if synchronization is needed
    /// based on cursor positions.
    fn analyse_shared_bindings<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> Vec<StreamId> {
        let mut to_align = Vec::new();
        let current = self.streams.get(&stream_id).unwrap();

        for binding in bindings {
            if stream_id != binding.stream {
                if to_align.contains(&binding.stream) {
                    continue;
                }

                if let Some(last_synced) = current.last_synced.get(&binding.stream) {
                    if *last_synced < binding.cursor {
                        to_align.push(binding.stream);
                    }
                } else {
                    to_align.push(binding.stream);
                }
            }
        }

        to_align
    }
}
