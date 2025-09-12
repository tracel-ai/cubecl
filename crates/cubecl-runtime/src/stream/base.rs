use crate::server::Bindings;
use cubecl_common::stream_id::StreamId;
use hashbrown::HashMap;

pub trait StreamBackend {
    type Stream;
    type Event;

    fn init_stream(stream_id: StreamId) -> Self::Stream;
    fn flush(stream: &mut Self::Stream) -> Self::Event;
    fn wait_event(stream: &mut Self::Stream, event: Self::Event);
}

#[derive(Default)]
pub struct MultiStream<B: StreamBackend> {
    streams: HashMap<StreamId, StreamWrapper<B>>,
}

pub struct StreamWrapper<B: StreamBackend> {
    stream: B::Stream,
    last_synced: HashMap<StreamId, u64>,
    cursor: u64,
}

impl<B: StreamBackend> MultiStream<B> {
    pub fn resolve(&mut self, stream_id: StreamId, bindings: &Bindings) -> &'_ mut B::Stream {
        self.align_streams(stream_id, bindings);

        &mut self
            .streams
            .get_mut(&stream_id)
            .expect("Stream to exist")
            .stream
    }

    fn align_streams(&mut self, stream_id: StreamId, bindings: &Bindings) {
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

        let mut events = Vec::with_capacity(analyses.len());

        for stream_origin in analyses {
            let stream = self.streams.get_mut(&stream_origin).unwrap();
            let event = B::flush(&mut stream.stream);

            events.push(((stream_origin, stream.cursor), event));
        }

        let stream = self.streams.get_mut(&stream_id).unwrap();

        for ((stream_origin, cursor_origin), event) in events {
            stream.last_synced.insert(stream_origin, cursor_origin);

            B::wait_event(&mut stream.stream, event);
        }
    }

    fn analyse_shared_bindings(
        &mut self,
        stream_id: StreamId,
        bindings: &Bindings,
    ) -> Vec<StreamId> {
        let mut to_align = Vec::new();
        let current = self.streams.get(&stream_id).unwrap();

        for binding in bindings.buffers.iter() {
            if stream_id != binding.stream {
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
