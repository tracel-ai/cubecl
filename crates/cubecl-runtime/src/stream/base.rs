use alloc::vec::Vec;
use cubecl_common::stream_id::StreamId;

/// Trait for creating streams, used by the stream pool to generate streams as needed.
pub trait StreamFactory {
    /// The type of stream produced by this factory.
    type Stream;
    /// Creates a new stream instance.
    fn create(&mut self) -> Self::Stream;
}

/// Represents a pool of streams, managing a collection of streams created by a factory.
#[derive(Debug)]
pub struct StreamPool<F: StreamFactory> {
    /// Vector storing optional streams, where None indicates an uninitialized stream.
    streams: Vec<Option<F::Stream>>,
    /// The factory used to create new streams when needed.
    factory: F,
    /// Maximum number of regular streams (excludes special streams).
    max_streams: usize,
}

impl<F: StreamFactory> StreamPool<F> {
    /// Creates a new stream pool with the given backend factory and capacity constraints.
    pub fn new(backend: F, max_streams: u8, num_special: u8) -> Self {
        // Initialize a vector with capacity for regular and special streams.
        let mut streams = Vec::with_capacity(max_streams as usize);
        // Pre-populate the vector with None to reserve space for all streams.
        for _ in 0..(max_streams + num_special) {
            streams.push(None);
        }

        Self {
            streams,
            factory: backend,
            max_streams: max_streams as usize,
        }
    }

    /// Retrieves a mutable reference to a stream for a given stream ID.
    pub fn get_mut(&mut self, stream_id: &StreamId) -> &mut F::Stream {
        // Calculate the index for the stream ID.
        let index = self.stream_index(stream_id);

        // Use unsafe method to retrieve the stream, assuming the index is valid.
        //
        // # Safety
        //
        // * The `stream_index` function ensures the index is within bounds.
        unsafe { self.get_mut_index(index) }
    }

    /// Retrieves a mutable reference to a stream at the specified index, initializing it if needed.
    ///
    /// # Safety
    ///
    /// * Caller must ensure the index is valid (less than `max_streams + num_special`).
    /// * Lifetimes still follow the Rust rules.
    pub unsafe fn get_mut_index(&mut self, index: usize) -> &mut F::Stream {
        unsafe {
            // Access the stream entry without bounds checking for performance.
            let entry = self.streams.get_unchecked_mut(index);
            match entry {
                // If the stream exists, return it.
                Some(val) => val,
                // If the stream is None, create a new one using the factory.
                None => {
                    let stream = self.factory.create();
                    // Store the new stream in the vector.
                    *entry = Some(stream);

                    // Re-access the entry, which is now guaranteed to be Some.
                    match entry {
                        Some(val) => val,
                        // Unreachable because we just set it to Some.
                        None => unreachable!(),
                    }
                }
            }
        }
    }

    /// Retrieves a mutable reference to a special stream at the given index.
    ///
    /// # Safety
    ///
    /// * Caller must ensure the index corresponds to a valid special stream.
    /// * Lifetimes still follow the Rust rules.
    pub unsafe fn get_special(&mut self, index: u8) -> &mut F::Stream {
        // Calculate the index for the special stream (offset by max_streams).
        unsafe { self.get_mut_index(self.max_streams + index as usize) }
    }

    /// Calculates the index for a given stream ID, mapping it to the pool's capacity.
    pub fn stream_index(&mut self, id: &StreamId) -> usize {
        stream_index(id, self.max_streams)
    }
}

/// Maps a stream ID to an index within the pool's capacity using modulo arithmetic.
pub fn stream_index(stream_id: &StreamId, max_streams: usize) -> usize {
    stream_id.value as usize % max_streams
}
