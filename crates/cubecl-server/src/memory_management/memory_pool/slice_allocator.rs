use super::{ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice};
use crate::{
    memory_management::{ErrorGraph, MemoryLocation},
    server::IoError,
    storage::{StorageAllocator, StorageHandle, StorageId, StorageUtilization},
};
use cubecl_environment::backtrace::BackTrace;

/// Allocates, binds, hands over and releases the slices of the pools: the
/// steps every pool takes the same way, whatever it carves.
///
/// # Assumptions
///
/// `failures`, in every function that takes it, is the [`ErrorGraph`] that
/// holds the failures the slices carry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceAllocator;

impl SliceAllocator {
    /// A storage handle for `size` bytes honoring `mapping`: a real device
    /// allocation when [`Eager`](PageMapping::Eager), a minted id awaiting
    /// backing when [`Lazy`](PageMapping::Lazy).
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses, for an eager mapping.
    pub(crate) fn allocate_storage<Storage: StorageAllocator>(
        &self,
        storage: &mut Storage,
        mapping: PageMapping,
        size: u64,
    ) -> Result<StorageHandle, IoError> {
        match mapping {
            PageMapping::Eager => storage.allocate_bytes(size),
            PageMapping::Lazy => Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: size,
                },
            )),
        }
    }

    /// A slice that owns its whole device allocation of `padded_size` bytes,
    /// serving `size` of them at `location`, backed now or later as `mapping`
    /// says.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses.
    pub(crate) fn allocate_whole<Storage: StorageAllocator>(
        &self,
        storage: &mut Storage,
        size: u64,
        padded_size: u64,
        mapping: PageMapping,
        location: MemoryLocation,
    ) -> Result<Slice, IoError> {
        let mut slice = Slice::new(self.allocate_storage(storage, mapping, padded_size)?, 0);
        slice.mapped = mapping == PageMapping::Eager;
        self.fit_whole(&mut slice, size, padded_size);
        slice.descriptor().update_location(location);
        Ok(slice)
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

    /// Give a slice that owns its whole allocation, found for `binding`, real
    /// device backing if it was allocated [`PageMapping::Lazy`]: allocate for
    /// real and retire the minted id, which never reached the driver.
    ///
    /// A missing slice, a mapped one and one `binding` has no claim on are left
    /// alone: lookup errors are the lookup's to report, and backing a slice
    /// nobody asked to resolve would allocate device memory for nothing.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the allocation.
    pub(crate) fn materialize_whole<Storage: StorageAllocator>(
        &self,
        storage: &mut Storage,
        slice: Option<&mut Slice>,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let Some(slice) =
            slice.filter(|slice| !slice.mapped && slice.descriptor() == binding.descriptor())
        else {
            return Ok(());
        };
        let size = slice.effective_size();
        let real = storage
            .allocate_bytes(size)
            .map_err(|err| IoError::StorageMappingFailed {
                size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })?;
        slice.storage.id = real.id;
        slice.mapped = true;
        Ok(())
    }

    /// Release a slice that owns its whole allocation: the failure it carried
    /// goes with it, and its allocation goes back to the driver unless it was
    /// never backed, in which case there is nothing behind its minted id.
    pub(crate) fn release_whole<Storage: StorageAllocator>(
        &self,
        storage: &mut Storage,
        slice: &mut Slice,
        failures: &mut ErrorGraph,
    ) {
        slice.tainted.clear(failures);
        if slice.mapped {
            storage.deallocate(slice.storage.id);
        }
    }

    /// Bind `assigned` to the allocation on `slice` at `cursor`: whatever
    /// tainted the last allocation says nothing about the next, so every claim
    /// is released before the handle changes hands.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerError;
    use crate::storage::BytesStorage;

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

    fn failure(failures: &mut ErrorGraph) -> crate::memory_management::FailureId {
        failures.insert(ServerError::Generic {
            reason: "launch".into(),
            backtrace: Default::default(),
        })
    }

    /// A lazy storage is a minted id the storage never allocated.
    #[test]
    fn a_lazy_storage_is_not_allocated() {
        let mut storage = BytesStorage::default();

        let lazy = SliceAllocator
            .allocate_storage(&mut storage, PageMapping::Lazy, 64)
            .expect("a minted id should never fail");

        assert_eq!((lazy.size(), storage.bytes_allocated()), (64, 0));
    }

    /// An eager storage is allocated now.
    #[test]
    fn an_eager_storage_is_allocated_now() {
        let mut storage = BytesStorage::default();

        SliceAllocator
            .allocate_storage(&mut storage, PageMapping::Eager, 64)
            .expect("the host should have 64 bytes");

        assert_eq!(storage.bytes_allocated(), 64);
    }

    /// A whole slice serves its size at its location, padded to the rest.
    #[test]
    fn a_whole_slice_serves_its_size_at_its_location() {
        let mut storage = BytesStorage::default();

        let whole = SliceAllocator
            .allocate_whole(
                &mut storage,
                100,
                128,
                PageMapping::Lazy,
                MemoryLocation::new(1, 0, 3),
            )
            .expect("a lazy slice should never fail");

        assert_eq!((whole.storage.size(), whole.padding_bytes), (100, 28));
        assert!(!whole.mapped);
        assert_eq!(whole.descriptor().location(), MemoryLocation::new(1, 0, 3));
    }

    /// A missing slice, and one a foreign binding names, are left unbacked.
    #[test]
    fn a_missing_or_foreign_slice_is_left_unbacked() {
        let mut storage = BytesStorage::default();
        let mut owned = slice(64);
        owned.mapped = false;

        SliceAllocator
            .materialize_whole(&mut storage, None, &owned.handle.clone().binding())
            .expect("nothing should be backed");
        SliceAllocator
            .materialize_whole(
                &mut storage,
                Some(&mut owned),
                &ManagedMemoryHandle::new().binding(),
            )
            .expect("a foreign binding should be left alone");

        assert!(!owned.mapped);
        assert_eq!(storage.bytes_allocated(), 0);
    }

    /// An unmapped slice the binding owns is backed.
    #[test]
    fn an_owned_unmapped_slice_is_backed() {
        let mut storage = BytesStorage::default();
        let mut owned = slice(64);
        owned.mapped = false;
        let binding = owned.handle.clone().binding();

        SliceAllocator
            .materialize_whole(&mut storage, Some(&mut owned), &binding)
            .expect("the storage should back the slice");

        assert!(owned.mapped);
        assert_eq!(storage.bytes_allocated(), 64);
    }

    /// A released slice returns its allocation only when it was backed.
    #[test]
    fn a_released_slice_returns_only_a_backed_allocation() {
        let mut storage = BytesStorage::default();
        let mut failures = ErrorGraph::default();
        let mut backed = Slice::new(
            storage
                .allocate_bytes(64)
                .expect("the host should have 64 bytes"),
            0,
        );
        let mut minted = slice(64);
        minted.mapped = false;

        SliceAllocator.release_whole(&mut storage, &mut minted, &mut failures);
        SliceAllocator.release_whole(&mut storage, &mut backed, &mut failures);
        storage.flush();

        assert_eq!(storage.bytes_allocated(), 0);
    }

    /// A binding takes the slice's location and cursor, and the failures the
    /// last allocation carried go.
    #[test]
    fn a_binding_releases_the_last_allocations_failures() {
        let mut failures = ErrorGraph::default();
        let mut slice = slice(64);
        slice
            .descriptor()
            .update_location(MemoryLocation::new(3, 1, 2));
        let failure = failure(&mut failures);
        slice.tainted.taint(0..64, failure, &mut failures);
        let assigned = ManagedMemoryHandle::new();

        SliceAllocator.bind(&mut slice, assigned.clone(), 5, &mut failures);

        assert_eq!(
            assigned.descriptor().location(),
            MemoryLocation::new(3, 1, 2)
        );
        assert_eq!(slice.cursor, 5);
        assert!(failures.is_empty());
    }

    /// A hand-over moves the allocation, its cursor and its taint to the
    /// target, and leaves the source free.
    #[test]
    fn a_hand_over_moves_the_allocation_and_its_taint() {
        let mut failures = ErrorGraph::default();
        let mut source = slice(64);
        let mut target = slice(64);
        target
            .descriptor()
            .update_location(MemoryLocation::new(4, 0, 1));
        let owner = source.handle.clone();
        source.cursor = 9;
        let failure = failure(&mut failures);
        source.tainted.taint(0..64, failure, &mut failures);

        SliceAllocator.hand_over(&mut source, &mut target, &mut failures);

        assert_eq!(owner.descriptor().location(), MemoryLocation::new(4, 0, 1));
        assert_eq!(
            (target.cursor, target.tainted.failure(&(0..64))),
            (9, Some(failure))
        );
        assert!(source.is_free() && source.tainted.is_clean());
    }
}
