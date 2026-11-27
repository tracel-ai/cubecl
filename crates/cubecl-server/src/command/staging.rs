//! What a host-to-device copy should do about its source buffer.
//!
//! The decision is plain data with a constructor, which is the point: it is
//! the only part of a copy that can be checked without a device, and inline in
//! the copy it never was.

use crate::memory_management::drop_queue::StreamSyncer;
use crate::server::ServerError;
use cubecl_common::bytes::{AllocationProperty, Bytes};

/// A megabyte, for the thresholds below.
const MB: usize = 1024 * 1024;

/// Transfers up to this size go through a pinned staging buffer, which the
/// driver can DMA from without a bounce. Above it, pinning a buffer the size
/// of the whole copy costs more than it saves, so the copy is either chunked
/// or handed over as it is.
const STAGE_MAX: usize = 100 * MB;

/// Above this size the drop queue is flushed after the copy, so the source is
/// released promptly rather than waiting for the next batch to fill.
const FLUSH_MIN: usize = 10 * MB;

/// What a host-to-device copy does about its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staging {
    /// How the source reaches the driver.
    pub kind: StagingKind,
    /// Flush the drop queue once the copy is enqueued, rather than leaving the
    /// source held until the next batch fills.
    pub flush_after: bool,
}

/// How the source of a host-to-device copy reaches the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingKind {
    /// The source is handed to the driver as it is.
    Direct,
    /// The source is copied whole into one pinned buffer first.
    Whole,
    /// The source is copied through a [`ChunkStager`], one pinned chunk at a
    /// time, each chunk's host copy overlapping the device's read of the one
    /// before.
    Chunked,
}

impl Staging {
    /// The size of each chunk a [`Chunked`](StagingKind::Chunked) copy is
    /// split into.
    pub const CHUNK_BYTES: usize = 8 * MB;

    /// What a copy of `size` bytes of memory allocated as `property` should
    /// do, where `chunkable` says whether the copy may be split into linear
    /// chunks: its driver gains from it, its target is one contiguous span,
    /// and no graph is recording it.
    pub fn of(size: usize, property: AllocationProperty, chunkable: bool) -> Self {
        let file_backed = matches!(property, AllocationProperty::File);
        let pinned = matches!(property, AllocationProperty::Pinned);
        // File-backed data is staged whatever its size: the driver reads the
        // source asynchronously, and it has to be real memory by then.
        // Otherwise stage whole only what is small enough to be worth pinning,
        // and never what is pinned already, which would be a redundant
        // pinned-to-pinned copy.
        let whole = file_backed || (size < STAGE_MAX && !pinned);
        let chunked = chunkable && !whole && !pinned;
        let kind = match (whole, chunked) {
            (true, _) => StagingKind::Whole,
            (false, true) => StagingKind::Chunked,
            (false, false) => StagingKind::Direct,
        };
        Self {
            kind,
            // A large source, or one mapped from a file, is worth releasing
            // now rather than holding until the batch fills.
            flush_after: file_backed || size > FLUSH_MIN,
        }
    }
}

/// Bounces a large copy through two pinned buffers, so the host fills one
/// while the device reads the other.
///
/// A copy from pageable memory makes the driver stage it itself, serially and
/// at a fraction of the bandwidth a pinned source reaches. Pinning a buffer
/// the size of the whole copy instead costs time in proportion to its size,
/// and holds that much pinned memory.
///
/// # Assumptions
///
/// Every fence handed to [`fence`](Self::fence) is recorded on the stream that
/// reads the chunk [`stage`](Self::stage) last returned, after the copy of it
/// was enqueued.
///
/// # Guarantees
///
/// - A buffer is overwritten only once the fence recorded after the copy of
///   its last chunk has been reached.
/// - Two buffers of [`Staging::CHUNK_BYTES`] are held, whatever the
///   size of the copy.
pub struct ChunkStager<F: StreamSyncer> {
    /// The buffers, each with the fence after the copy of its last chunk.
    /// Position Usage: the buffer at `next` is filled next.
    slots: [(Bytes, Option<F>); 2],
    next: usize,
}

impl<F: StreamSyncer> ChunkStager<F> {
    /// A stager over two pinned buffers.
    ///
    /// # Assumptions
    ///
    /// Both buffers hold at least [`Staging::CHUNK_BYTES`].
    pub fn new(first: Bytes, second: Bytes) -> Self {
        Self {
            slots: [(first, None), (second, None)],
            next: 0,
        }
    }

    /// Copy `chunk` into the next buffer, once the device has finished
    /// reading what that buffer held, and return the copy.
    ///
    /// # Assumptions
    ///
    /// `chunk` holds at most [`Staging::CHUNK_BYTES`].
    ///
    /// # Panics
    ///
    /// When `chunk` is longer than the buffer, which breaks this Assumption or
    /// the one of [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// The fault waiting on the buffer's fence reveals.
    pub fn stage(&mut self, chunk: &[u8]) -> Result<&[u8], ServerError> {
        let (buffer, fence) = &mut self.slots[self.next];
        if let Some(fence) = fence.take() {
            fence.sync_host()?;
        }
        let staged = &mut buffer[..chunk.len()];
        staged.copy_from_slice(chunk);
        Ok(staged)
    }

    /// Record `fence` as the one after the copy of the chunk
    /// [`stage`](Self::stage) last returned, and move on to the other buffer.
    pub fn fence(&mut self, fence: F) {
        self.slots[self.next].1 = Some(fence);
        self.next = 1 - self.next;
    }

    /// The buffers, for the caller to keep alive until the device has read
    /// them.
    pub fn into_buffers(self) -> [Bytes; 2] {
        self.slots.map(|(buffer, _)| buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec;
    use core::cell::Cell;

    /// File-backed data is staged whole whatever its size, chunkable or not.
    ///
    /// The driver reads the source after the copy is enqueued, so a mapping
    /// that may still fault has to become real memory first. Size is why
    /// staging is *worth* it for everything else; for a file it is why the
    /// copy is correct.
    #[test]
    fn file_backed_data_is_always_staged_whole() {
        for size in [1, STAGE_MAX, STAGE_MAX * 4] {
            let staging = Staging::of(size, AllocationProperty::File, true);
            assert_eq!(staging.kind, StagingKind::Whole);
        }
    }

    /// Already-pinned memory is handed to the driver as it is.
    ///
    /// Staging it, whole or in chunks, would copy pinned memory into pinned
    /// memory, paying for a bounce that exists to avoid one.
    #[test]
    fn pinned_data_is_never_restaged() {
        for size in [1, STAGE_MAX / 2, STAGE_MAX * 4] {
            let staging = Staging::of(size, AllocationProperty::Pinned, true);
            assert_eq!(staging.kind, StagingKind::Direct);
        }
    }

    /// Ordinary host memory is staged whole only while it is small enough
    /// that pinning costs less than the bounce it saves.
    #[test]
    fn plain_data_is_staged_whole_up_to_the_threshold() {
        let below = Staging::of(STAGE_MAX - 1, AllocationProperty::Native, true);
        assert_eq!(below.kind, StagingKind::Whole);
        let at = Staging::of(STAGE_MAX, AllocationProperty::Native, false);
        assert_eq!(at.kind, StagingKind::Direct);
    }

    /// Ordinary host memory past the threshold is chunked when the copy can
    /// be.
    #[test]
    fn large_plain_data_is_chunked_when_chunkable() {
        let staging = Staging::of(STAGE_MAX, AllocationProperty::Native, true);
        assert_eq!(staging.kind, StagingKind::Chunked);
    }

    /// A copy big enough to be worth releasing promptly flushes the queue,
    /// whether or not it was staged.
    #[test]
    fn a_large_source_is_released_without_waiting_for_the_batch() {
        assert!(!Staging::of(FLUSH_MIN, AllocationProperty::Native, false).flush_after);
        assert!(Staging::of(FLUSH_MIN + 1, AllocationProperty::Native, false).flush_after);
        assert!(Staging::of(1, AllocationProperty::File, false).flush_after);
    }

    /// A fence that counts how many times the host waited on it.
    struct CountingFence(Rc<Cell<usize>>);

    impl StreamSyncer for CountingFence {
        fn sync_host(self) -> Result<(), ServerError> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }

    fn stager() -> ChunkStager<CountingFence> {
        ChunkStager::new(
            Bytes::from_bytes_vec(vec![0; 4]),
            Bytes::from_bytes_vec(vec![0; 4]),
        )
    }

    /// A buffer is refilled only after the host waited on the fence of the
    /// copy that read it, and the other buffer is filled meanwhile.
    #[test]
    fn a_buffer_is_refilled_only_after_its_fence() {
        let waits = Rc::new(Cell::new(0));
        let mut stager = stager();

        stager.stage(&[1]).unwrap();
        stager.fence(CountingFence(waits.clone()));
        stager.stage(&[2]).unwrap();
        stager.fence(CountingFence(waits.clone()));
        assert_eq!(waits.get(), 0);

        assert_eq!(stager.stage(&[3]).unwrap(), &[3]);
        assert_eq!(waits.get(), 1);
    }

    /// The buffers come back holding the last chunk each was given.
    #[test]
    fn the_buffers_hold_the_chunks_staged_into_them() {
        let mut stager = stager();
        stager.stage(&[1, 2]).unwrap();
        stager.fence(CountingFence(Rc::new(Cell::new(0))));
        stager.stage(&[3]).unwrap();

        let [first, second] = stager.into_buffers();
        assert_eq!(&first[..2], &[1, 2]);
        assert_eq!(second[0], 3);
    }
}
