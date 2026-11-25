use super::{ManagedMemoryBinding, PageMapping, Slice, SliceBinder};
use crate::{
    memory_management::{ErrorGraph, MemoryLocation},
    server::IoError,
    storage::{StorageAllocator, StorageHandle, StorageId, StorageUtilization},
};
use cubecl_environment::backtrace::BackTrace;

/// Allocates the storage of the pools' slices from the storage it holds,
/// backs it when it was allocated lazily, and releases it.
///
/// A reserver builds one over its storage for the steps that touch it.
pub(crate) struct SliceAllocator<'s, Storage> {
    storage: &'s mut Storage,
}

impl<'s, Storage: StorageAllocator> SliceAllocator<'s, Storage> {
    /// An allocator over `storage`.
    pub(crate) fn new(storage: &'s mut Storage) -> Self {
        Self { storage }
    }

    /// A storage handle for `size` bytes honoring `mapping`: a real device
    /// allocation when [`Eager`](PageMapping::Eager), a minted id awaiting
    /// backing when [`Lazy`](PageMapping::Lazy).
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses, for an eager mapping.
    pub(crate) fn allocate_storage(
        &mut self,
        mapping: PageMapping,
        size: u64,
    ) -> Result<StorageHandle, IoError> {
        match mapping {
            PageMapping::Eager => self.storage.allocate_bytes(size),
            PageMapping::Lazy => Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: size,
                },
            )),
        }
    }

    /// The real device allocation of `size` bytes that backs storage
    /// allocated lazily.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses it.
    pub(crate) fn allocate_backing(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        self.storage
            .allocate_bytes(size)
            .map_err(|err| IoError::StorageMappingFailed {
                size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })
    }

    /// A slice that owns its whole device allocation of `padded_size` bytes,
    /// serving `size` of them at `location`, backed now or later as `mapping`
    /// says.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses.
    pub(crate) fn allocate_whole(
        &mut self,
        size: u64,
        padded_size: u64,
        mapping: PageMapping,
        location: MemoryLocation,
    ) -> Result<Slice, IoError> {
        let mut slice = Slice::new(self.allocate_storage(mapping, padded_size)?, 0);
        slice.mapped = mapping == PageMapping::Eager;
        SliceBinder.fit_whole(&mut slice, size, padded_size);
        slice.descriptor().update_location(location);
        Ok(slice)
    }

    /// Give a slice that owns its whole allocation, found for `binding`, real
    /// device backing if it was allocated [`PageMapping::Lazy`]: allocate for
    /// real and retire the minted id, which never reached the driver.
    ///
    /// A missing slice, a mapped one and one `binding` has no claim on are
    /// left alone: lookup errors are the lookup's to report, and backing a
    /// slice nobody asked to resolve would allocate device memory for nothing.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the allocation.
    pub(crate) fn materialize_whole(
        &mut self,
        slice: Option<&mut Slice>,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let Some(slice) = slice else {
            return Ok(());
        };
        if slice.mapped || slice.descriptor() != binding.descriptor() {
            return Ok(());
        }
        slice.storage.id = self.allocate_backing(slice.effective_size())?.id;
        slice.mapped = true;
        Ok(())
    }

    /// Release a slice that owns its whole allocation: the failure it carried
    /// goes with it, and its allocation goes back to the driver unless it was
    /// never backed, in which case there is nothing behind its minted id.
    ///
    /// # Panics
    ///
    /// When `failures` does not hold the failures the slice carries.
    pub(crate) fn release_whole(&mut self, slice: &mut Slice, failures: &mut ErrorGraph) {
        slice.tainted.clear(failures);
        if slice.mapped {
            self.storage.deallocate(slice.storage.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;
    use crate::memory_management::memory_pool::CountingStorageAllocator;
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

    /// A lazy storage is a minted id the storage never allocated.
    #[test]
    fn a_lazy_storage_is_not_allocated() {
        let mut storage = BytesStorage::default();

        let lazy = SliceAllocator::new(&mut storage)
            .allocate_storage(PageMapping::Lazy, 64)
            .expect("a minted id should never fail");

        assert_eq!((lazy.size(), storage.bytes_allocated()), (64, 0));
    }

    /// An eager storage is allocated now.
    #[test]
    fn an_eager_storage_is_allocated_now() {
        let mut storage = BytesStorage::default();

        SliceAllocator::new(&mut storage)
            .allocate_storage(PageMapping::Eager, 64)
            .expect("the host should have 64 bytes");

        assert_eq!(storage.bytes_allocated(), 64);
    }

    /// A whole slice serves its size, padded to the rest.
    #[test]
    fn a_whole_slice_serves_its_size_padded() {
        let mut storage = BytesStorage::default();

        let whole = SliceAllocator::new(&mut storage)
            .allocate_whole(100, 128, PageMapping::Lazy, MemoryLocation::new(1, 0, 3))
            .expect("a lazy slice should never fail");

        assert_eq!((whole.storage.size(), whole.padding_bytes), (100, 28));
    }

    /// A whole slice allocated lazily is not mapped.
    #[test]
    fn a_lazy_whole_slice_is_unmapped() {
        let mut storage = BytesStorage::default();

        let whole = SliceAllocator::new(&mut storage)
            .allocate_whole(100, 128, PageMapping::Lazy, MemoryLocation::new(1, 0, 3))
            .expect("a lazy slice should never fail");

        assert!(!whole.mapped);
    }

    /// A whole slice is placed at its location.
    #[test]
    fn a_whole_slice_is_placed_at_its_location() {
        let mut storage = BytesStorage::default();

        let whole = SliceAllocator::new(&mut storage)
            .allocate_whole(100, 128, PageMapping::Lazy, MemoryLocation::new(1, 0, 3))
            .expect("a lazy slice should never fail");

        assert_eq!(whole.descriptor().location(), MemoryLocation::new(1, 0, 3));
    }

    /// A missing slice is left unbacked.
    #[test]
    fn a_missing_slice_is_left_unbacked() {
        let mut storage = BytesStorage::default();

        SliceAllocator::new(&mut storage)
            .materialize_whole(None, &ManagedMemoryHandle::new().binding())
            .expect("nothing should be backed");

        assert_eq!(storage.bytes_allocated(), 0);
    }

    /// A slice a foreign binding names is left unbacked.
    #[test]
    fn a_foreign_slice_is_left_unbacked() {
        let mut storage = BytesStorage::default();
        let mut owned = slice(64);
        owned.mapped = false;

        SliceAllocator::new(&mut storage)
            .materialize_whole(Some(&mut owned), &ManagedMemoryHandle::new().binding())
            .expect("a foreign binding should be left alone");

        assert!(!owned.mapped);
    }

    /// An unmapped slice the binding owns is backed.
    #[test]
    fn an_owned_unmapped_slice_is_backed() {
        let mut storage = BytesStorage::default();
        let mut owned = slice(64);
        owned.mapped = false;
        let binding = owned.handle.clone().binding();

        SliceAllocator::new(&mut storage)
            .materialize_whole(Some(&mut owned), &binding)
            .expect("the storage should back the slice");

        assert!(owned.mapped);
        assert_eq!(storage.bytes_allocated(), 64);
    }

    /// A released slice that was backed goes back to the driver.
    #[test]
    fn a_released_backed_slice_is_deallocated() {
        let mut storage = CountingStorageAllocator::default();
        let mut failures = ErrorGraph::default();
        let mut backed = Slice::new(
            storage
                .allocate_bytes(64)
                .expect("the host should have 64 bytes"),
            0,
        );
        let id = backed.storage.id;

        SliceAllocator::new(&mut storage).release_whole(&mut backed, &mut failures);

        assert_eq!(storage.deallocated(), [id]);
    }

    /// A released slice never backed hands the driver nothing.
    #[test]
    fn a_released_minted_slice_is_not_deallocated() {
        let mut storage = CountingStorageAllocator::default();
        let mut failures = ErrorGraph::default();
        let mut minted = slice(64);
        minted.mapped = false;

        SliceAllocator::new(&mut storage).release_whole(&mut minted, &mut failures);

        assert!(storage.deallocated().is_empty());
    }
}
