use crate::memory_management::{ErrorGraph, FailureId, ManagedMemoryBinding};
use crate::server::{BufferBinding, ServerError};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;

/// Trait for creating streams, used by the stream pool to generate streams as needed.
pub trait StreamFactory {
    /// The type of stream produced by this factory.
    type Stream;
    /// Creates a new stream instance.
    fn create(&mut self) -> Self::Stream;
}

/// The memory a stream's kernels see, for the taint bookkeeping.
///
/// Whether a buffer can be trusted lives on its allocation, inside one of the
/// stream's memory managers. A backend supplies only which manager that is —
/// the one whose allocations back [`BufferBinding`]s, never the auxiliary
/// staging or uniform managers — and everything the drivers do with the
/// answer lives on the shared wrappers.
pub trait StreamMemory {
    /// The failure carried by the allocation behind `binding`, if any.
    fn failure(&self, binding: &ManagedMemoryBinding) -> Option<FailureId>;

    /// Point the allocation behind `binding` at `failure`.
    fn taint(
        &mut self,
        binding: &ManagedMemoryBinding,
        failure: FailureId,
        failures: &mut ErrorGraph,
    );

    /// The allocation behind `binding` has a writer again: release the
    /// failure it carried.
    fn written(&mut self, binding: &ManagedMemoryBinding, failures: &mut ErrorGraph);
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
        for _ in 0..(max_streams.saturating_add(num_special)) {
            streams.push(None);
        }

        Self {
            streams,
            factory: backend,
            max_streams: max_streams as usize,
        }
    }

    /// Read-only iterator over initialized streams (unlike [`Self::get_mut`], never creates one).
    pub fn streams(&self) -> impl Iterator<Item = &F::Stream> {
        self.streams.iter().flatten()
    }

    /// Synthetic [`StreamId`]s, one per initialized regular pool slot.
    ///
    /// Each id round-trips through [`Self::get_mut`] to the same slot it
    /// came from (slot `i` is reachable via `StreamId { value: i }` since
    /// indexing is `value % max_streams`), so it's safe to feed these
    /// ids back into per-stream APIs.
    pub fn stream_ids(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.streams[..self.max_streams]
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| StreamId { value: i as u64 }))
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

    /// The stream on `id`'s slot, when that slot was ever initialized.
    ///
    /// Never creates one: resolving a buffer's owning slot must not bring a
    /// backend stream into existence, which on CUDA and HIP would bind it to
    /// whichever context happens to be current. A buffer's slot was
    /// initialized by the allocation itself, so `None` here means the binding
    /// is not this pool's to answer for.
    pub fn try_get(&self, id: &StreamId) -> Option<&F::Stream> {
        self.streams[stream_index(id, self.max_streams)].as_ref()
    }

    /// [`try_get`](Self::try_get), mutably.
    pub fn try_get_mut(&mut self, id: &StreamId) -> Option<&mut F::Stream> {
        self.streams[stream_index(id, self.max_streams)].as_mut()
    }

    /// Fails when the buffers `handles` name carry a failure, with the errors
    /// of the work that was supposed to write them.
    ///
    /// A read is only as good as the work that wrote the buffer: a launch that
    /// failed never wrote it, so copying its bytes out hands back whatever was
    /// in memory before. Whether that happened is a field on the allocation,
    /// so the question is answered by the slice each binding resolves to — a
    /// lookup the read was going to do anyway — and by nobody's queue.
    ///
    /// The errors are read, never taken: the stream that caused each one still
    /// surfaces it on its own flush.
    ///
    /// # Errors
    ///
    /// [`ServerError::ServerUnhealthy`] naming every failure one of these
    /// buffers carries, each failure once however many buffers carry it. The
    /// caller has nothing to retry — the bytes are gone — so the error is the
    /// answer to the read, not a hint to try again.
    pub fn ensure_written<'a>(
        &self,
        failures: &ErrorGraph,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError>
    where
        F::Stream: StreamMemory,
    {
        let mut seen: Vec<FailureId> = Vec::new();
        let mut errors = Vec::new();

        for handle in handles {
            let Some(stream) = self.try_get(&handle.stream) else {
                continue;
            };
            let Some(failure) = stream.failure(&handle.memory) else {
                continue;
            };
            if seen.contains(&failure) {
                continue;
            }
            seen.push(failure);
            if let Some(error) = failures.error(failure) {
                errors.push(error.clone());
            }
        }

        match errors.is_empty() {
            true => Ok(()),
            false => Err(ServerError::ServerUnhealthy {
                errors,
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// Taint every allocation in `written` with `error`: the work that was
    /// going to write those buffers did not run, so a read of any of them
    /// fails on this failure until something writes them again.
    ///
    /// Each binding is resolved to the manager of the stream it was created
    /// on, which may not be the stream that failed — that is the point: the
    /// fact lands on the memory, wherever it lives.
    pub fn taint<'a>(
        &mut self,
        error: ServerError,
        written: impl Iterator<Item = &'a BufferBinding>,
        failures: &mut ErrorGraph,
    ) where
        F::Stream: StreamMemory,
    {
        let failure = failures.insert(error);
        for handle in written {
            let index = stream_index(&handle.stream, self.max_streams);
            let Some(stream) = self.streams[index].as_mut() else {
                continue;
            };
            stream.taint(&handle.memory, failure, failures);
        }
        // A failure that named no buffer anything still holds has nothing to
        // wait for.
        failures.prune(failure);
    }

    /// Release the failure on every allocation in `written`: work that writes
    /// them has been enqueued, so a read of one is no longer reading bytes
    /// nothing wrote.
    pub fn written<'a>(
        &mut self,
        written: impl Iterator<Item = &'a BufferBinding>,
        failures: &mut ErrorGraph,
    ) where
        F::Stream: StreamMemory,
    {
        for handle in written {
            let index = stream_index(&handle.stream, self.max_streams);
            let Some(stream) = self.streams[index].as_mut() else {
                continue;
            };
            stream.written(&handle.memory, failures);
        }
    }

    /// Mutable access to the factory, e.g. to change the configuration new
    /// streams are created with. Already-created streams are unaffected.
    pub fn factory_mut(&mut self) -> &mut F {
        &mut self.factory
    }
}

/// Maps a stream ID to an index within the pool's capacity using modulo arithmetic.
pub fn stream_index(stream_id: &StreamId, max_streams: usize) -> usize {
    stream_id.value as usize % max_streams
}
