//! The pool of backend streams, and the primitives that record on their
//! memory what a unit of work did or did not write.
//!
//! The pool answers one question — which stream sits in a slot — and the
//! free functions below answer the other: which stream allocated a binding,
//! so the claim lands on the memory rather than on whoever failed. They are
//! free because the pool and the failure graph are held apart by every
//! caller; the surface a driver actually uses is
//! [`FailureStore`](super::FailureStore).

use crate::memory_management::{ErrorGraph, FailureId, ManagedMemoryId};
use crate::server::{BufferBinding, ServerError};
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// What a launch's read-set check found: the failure claiming an input, the
/// input it claims, and the error — everything the skip needs to record and
/// a capture needs to fail with.
pub struct ReadFailure {
    /// The failure claiming the input.
    pub failure: FailureId,
    /// The claimed input, which the skip record names as what the launch
    /// needed.
    pub needed: ManagedMemoryId,
    /// The failure's error, cloned for the paths that report it directly.
    pub error: ServerError,
}

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
///
/// The whole binding is passed rather than its memory handle because a
/// binding names a byte range of its allocation ([`BufferBinding::range`]),
/// and the claim is exactly that range: a launch that failed writing one
/// region of a buffer says nothing about the rest of it.
pub trait StreamMemory {
    /// The failure claiming any byte the binding names, if one does.
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId>;

    /// Point the bytes `binding` names at `failure`.
    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph);

    /// The bytes `binding` names have a writer again: release every claim on
    /// them, and only on them.
    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph);
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

/// Point the bytes every binding in `written` names at `failure`.
///
/// Each binding is resolved to the stream that allocated it, which may not be
/// the stream that failed — that is the point: the fact lands on the memory,
/// wherever it lives. A binding whose slot no stream ever initialized is
/// skipped; it is not this pool's to answer for.
///
/// Free rather than a method because the pool and the graph are held apart by
/// every caller: a driver owns both, a resolved borrow holds both mutably.
pub fn taint_with<'a, F>(
    pool: &mut StreamPool<F>,
    failure: FailureId,
    written: impl Iterator<Item = &'a BufferBinding>,
    graph: &mut ErrorGraph,
) where
    F: StreamFactory<Stream: StreamMemory>,
{
    for handle in written {
        if let Some(stream) = pool.try_get_mut(&handle.stream) {
            stream.taint(handle, failure, graph);
        }
    }
}

/// [`taint_with`] under a failure minted for `error`, dropped again when it
/// claimed nothing: a failure no buffer still holds has nothing to wait for.
pub fn taint<'a, F>(
    pool: &mut StreamPool<F>,
    error: ServerError,
    written: impl Iterator<Item = &'a BufferBinding>,
    graph: &mut ErrorGraph,
) where
    F: StreamFactory<Stream: StreamMemory>,
{
    let failure = graph.insert(error);
    taint_with(pool, failure, written, graph);
    graph.prune(failure);
}

/// Release the failure on every allocation in `written`: work that writes
/// them has been enqueued, so a read of one is no longer reading bytes
/// nothing wrote.
pub fn written<'a, F>(
    pool: &mut StreamPool<F>,
    written: impl Iterator<Item = &'a BufferBinding>,
    graph: &mut ErrorGraph,
) where
    F: StreamFactory<Stream: StreamMemory>,
{
    for handle in written {
        if let Some(stream) = pool.try_get_mut(&handle.stream) {
            stream.written(handle, graph);
        }
    }
}
