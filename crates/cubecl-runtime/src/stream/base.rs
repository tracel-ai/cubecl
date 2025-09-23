use cubecl_common::stream_id::StreamId;

pub trait StreamFactory {
    type Stream;
    fn create(&mut self) -> Self::Stream;
}

#[derive(Debug)]
pub struct StreamPool<F: StreamFactory> {
    streams: Vec<Option<F::Stream>>,
    factory: F,
    max_streams: usize,
}

impl<F: StreamFactory> StreamPool<F> {
    pub fn new(backend: F, max_streams: u8, num_special: u8) -> Self {
        let mut streams = Vec::with_capacity(max_streams as usize);
        for _ in 0..max_streams + num_special {
            streams.push(None);
        }

        Self {
            streams,
            factory: backend,
            max_streams: max_streams as usize,
        }
    }

    pub fn get_mut(&mut self, stream_id: &StreamId) -> &mut F::Stream {
        let index = self.stream_index(stream_id);

        // The pool is init, can't go over the index because of stream_index.
        unsafe { self.get_mut_index(index) }
    }

    pub unsafe fn get_mut_index(&mut self, index: usize) -> &mut F::Stream {
        unsafe {
            let entry = self.streams.get_unchecked_mut(index);
            match entry {
                Some(val) => val,
                None => {
                    let stream = self.factory.create();

                    *entry = Some(stream);

                    match entry {
                        Some(val) => val,
                        None => unreachable!(),
                    }
                }
            }
        }
    }

    pub unsafe fn get_special(&mut self, index: u8) -> &mut F::Stream {
        self.get_mut_index(self.max_streams + index as usize)
    }

    pub fn stream_index(&mut self, id: &StreamId) -> usize {
        stream_index(id, self.max_streams)
    }
}

pub fn stream_index(stream_id: &StreamId, max_streams: usize) -> usize {
    stream_id.value as usize % max_streams
}
