//! What a host-to-device copy should do about its source buffer.
//!
//! The decision is plain data with a constructor, which is the point: it is
//! the only part of a copy that can be checked without a device, and inline in
//! the copy it never was.

use cubecl_common::bytes::AllocationProperty;

/// A megabyte, for the thresholds below.
const MB: usize = 1024 * 1024;

/// Transfers up to this size go through a pinned staging buffer, which the
/// driver can DMA from without a bounce. Above it the copy is long enough that
/// the bounce costs less than pinning would.
const STAGE_MAX: usize = 100 * MB;

/// Above this size the drop queue is flushed after the copy, so the source is
/// released promptly rather than waiting for the next batch to fill.
const FLUSH_MIN: usize = 10 * MB;

/// What a host-to-device copy does about its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staging {
    /// Copy the source into pinned host memory before handing it to the
    /// driver.
    pub through_pinned: bool,
    /// Flush the drop queue once the copy is enqueued, rather than leaving the
    /// source held until the next batch fills.
    pub flush_after: bool,
}

impl Staging {
    /// What a copy of `size` bytes of memory allocated as `property` should do.
    pub fn of(size: usize, property: AllocationProperty) -> Self {
        let file_backed = matches!(property, AllocationProperty::File);
        Self {
            // File-backed data is staged whatever its size: the driver reads
            // the source asynchronously, and it has to be real memory by then.
            // Otherwise stage only what is small enough to be worth pinning,
            // and never what is pinned already — that would be a redundant
            // pinned-to-pinned copy.
            through_pinned: file_backed
                || (size < STAGE_MAX && !matches!(property, AllocationProperty::Pinned)),
            // A large source, or one mapped from a file, is worth releasing
            // now rather than holding until the batch fills.
            flush_after: file_backed || size > FLUSH_MIN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// File-backed data is staged whatever its size.
    ///
    /// The driver reads the source after the copy is enqueued, so a mapping
    /// that may still fault has to become real memory first. Size is why
    /// staging is *worth* it for everything else; for a file it is why the
    /// copy is correct.
    #[test]
    fn file_backed_data_is_always_staged() {
        for size in [1, STAGE_MAX, STAGE_MAX * 4] {
            assert!(Staging::of(size, AllocationProperty::File).through_pinned);
        }
    }

    /// Already-pinned memory is handed to the driver as it is.
    ///
    /// Staging it would copy pinned memory into pinned memory, paying for a
    /// bounce that exists to avoid one.
    #[test]
    fn pinned_data_is_never_restaged() {
        for size in [1, STAGE_MAX / 2, STAGE_MAX * 4] {
            assert!(!Staging::of(size, AllocationProperty::Pinned).through_pinned);
        }
    }

    /// Ordinary host memory is staged only while it is small enough that
    /// pinning costs less than the bounce it saves.
    #[test]
    fn plain_data_is_staged_up_to_the_threshold() {
        assert!(Staging::of(STAGE_MAX - 1, AllocationProperty::Native).through_pinned);
        assert!(!Staging::of(STAGE_MAX, AllocationProperty::Native).through_pinned);
    }

    /// A copy big enough to be worth releasing promptly flushes the queue,
    /// whether or not it was staged.
    #[test]
    fn a_large_source_is_released_without_waiting_for_the_batch() {
        assert!(!Staging::of(FLUSH_MIN, AllocationProperty::Native).flush_after);
        assert!(Staging::of(FLUSH_MIN + 1, AllocationProperty::Native).flush_after);
        assert!(Staging::of(1, AllocationProperty::File).flush_after);
    }
}
