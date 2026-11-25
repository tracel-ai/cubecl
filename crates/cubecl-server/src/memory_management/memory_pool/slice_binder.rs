use super::{ManagedMemoryHandle, Slice};
use crate::{memory_management::ErrorGraph, storage::StorageUtilization};

/// Binds allocations to slices, hands them from slice to slice, and fits a
/// slice to the allocation it serves: where the Commands on a [`Slice`] live,
/// since a slice is plain data the pools' pages and views hold.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceBinder;

impl SliceBinder {
    /// Bind `assigned` to the allocation on `slice` at `cursor`: whatever
    /// tainted the last allocation says nothing about the next, so every claim
    /// is released before the handle changes hands.
    ///
    /// # Assumptions
    ///
    /// `failures` holds the failures the slice carries.
    pub(crate) fn bind(
        &self,
        slice: &mut Slice,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) {
        assigned
            .descriptor()
            .update_location(slice.descriptor().location());
        slice.cursor = cursor;
        slice.tainted.clear(failures);
        slice.handle = assigned;
    }

    /// Give `source`'s allocation to `target`, which was reserved to receive
    /// it and already holds a copy of its bytes: the handle, its cursor and
    /// its taint move, the handle's location is rewritten to the target's so
    /// every owner resolves there from now on, and `source` is left free.
    /// Whatever the target's last allocation left tainted is released first.
    ///
    /// # Assumptions
    ///
    /// `failures` holds the failures the target carries.
    pub(crate) fn hand_over(
        &self,
        source: &mut Slice,
        target: &mut Slice,
        failures: &mut ErrorGraph,
    ) {
        target.tainted.clear(failures);
        let freed = ManagedMemoryHandle::new();
        freed
            .descriptor()
            .update_location(source.descriptor().location());
        let allocation = core::mem::replace(&mut source.handle, freed);
        allocation
            .descriptor()
            .update_location(target.descriptor().location());

        target.handle = allocation;
        target.cursor = source.cursor;
        core::mem::swap(&mut target.tainted, &mut source.tainted);
    }

    /// Serve `size` bytes from the start of a slice that owns its whole
    /// allocation, which `padded_size` bytes take once aligned.
    pub(crate) fn fit_whole(&self, slice: &mut Slice, size: u64, padded_size: u64) {
        slice.storage.utilization = StorageUtilization {
            offset_bytes: 0,
            size_bytes: size,
        };
        slice.padding_bytes = padded_size - size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::{FailureId, MemoryLocation};
    use crate::server::ServerError;
    use crate::storage::{StorageHandle, StorageId};

    fn slice(size: u64) -> Slice {
        Slice::new(
            StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: size,
                },
            ),
            0,
        )
    }

    fn failure(failures: &mut ErrorGraph) -> FailureId {
        failures.insert(ServerError::Generic {
            reason: "launch".into(),
            backtrace: Default::default(),
        })
    }

    /// A binding takes the slice's location.
    #[test]
    fn a_binding_takes_the_slices_location() {
        let mut slice = slice(64);
        slice
            .descriptor()
            .update_location(MemoryLocation::new(3, 1, 2));
        let assigned = ManagedMemoryHandle::new();

        SliceBinder.bind(&mut slice, assigned.clone(), 5, &mut ErrorGraph::default());

        assert_eq!(
            assigned.descriptor().location(),
            MemoryLocation::new(3, 1, 2)
        );
    }

    /// A binding takes the cursor it is bound at, which the cross-stream
    /// waits read.
    #[test]
    fn a_binding_takes_its_cursor() {
        let mut slice = slice(64);

        SliceBinder.bind(
            &mut slice,
            ManagedMemoryHandle::new(),
            5,
            &mut ErrorGraph::default(),
        );

        assert_eq!(slice.cursor, 5);
    }

    /// A binding releases the failures the last allocation carried.
    #[test]
    fn a_binding_releases_the_last_allocations_failures() {
        let mut failures = ErrorGraph::default();
        let mut slice = slice(64);
        let failure = failure(&mut failures);
        slice.tainted.taint(0..64, failure, &mut failures);

        SliceBinder.bind(&mut slice, ManagedMemoryHandle::new(), 0, &mut failures);

        assert!(failures.is_empty());
    }

    /// A hand-over moves the allocation to the target's location.
    #[test]
    fn a_hand_over_moves_the_allocation() {
        let mut source = slice(64);
        let mut target = slice(64);
        target
            .descriptor()
            .update_location(MemoryLocation::new(4, 0, 1));
        let owner = source.handle.clone();

        SliceBinder.hand_over(&mut source, &mut target, &mut ErrorGraph::default());

        assert_eq!(owner.descriptor().location(), MemoryLocation::new(4, 0, 1));
        assert!(source.is_free());
    }

    /// A hand-over moves the cursor and the taint with the allocation.
    #[test]
    fn a_hand_over_moves_the_cursor_and_the_taint_to_the_target() {
        let mut failures = ErrorGraph::default();
        let mut source = slice(64);
        let mut target = slice(64);
        let _owner = source.handle.clone();
        source.cursor = 9;
        let failure = failure(&mut failures);
        source.tainted.taint(0..64, failure, &mut failures);

        SliceBinder.hand_over(&mut source, &mut target, &mut failures);

        assert_eq!(
            (target.cursor, target.tainted.failure(&(0..64))),
            (9, Some(failure))
        );
    }

    /// A hand-over leaves the source with no claim.
    #[test]
    fn a_hand_over_leaves_the_source_clean() {
        let mut failures = ErrorGraph::default();
        let mut source = slice(64);
        let mut target = slice(64);
        let _owner = source.handle.clone();
        let failure = failure(&mut failures);
        source.tainted.taint(0..64, failure, &mut failures);

        SliceBinder.hand_over(&mut source, &mut target, &mut failures);

        assert!(source.tainted.is_clean());
    }

    /// Fitting a whole slice serves the size from its start, padded.
    #[test]
    fn fitting_a_whole_slice_serves_from_its_start() {
        let mut slice = slice(128);

        SliceBinder.fit_whole(&mut slice, 100, 128);

        assert_eq!(
            (
                slice.storage.offset(),
                slice.storage.size(),
                slice.padding_bytes
            ),
            (0, 100, 28)
        );
    }
}
