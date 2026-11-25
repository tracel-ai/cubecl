use super::{ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice, fit_whole_slice};
use crate::{
    memory_management::{Cleanup, ErrorGraph, MemoryLocation, MemoryPoolReport, PageGuard},
    server::IoError,
    storage::{StorageAllocator, StorageHandle, StorageId, StorageUtilization},
};
use cubecl_environment::backtrace::BackTrace;

/// Reserves slices of device memory for allocations, and reuses them once
/// their owners let go.
///
/// The functions that take a storage are for callers holding the pool's own
/// type; the rest can be reached through `dyn SliceReserver`, which is how a
/// location is resolved to whichever pool it names.
///
/// # Assumptions
///
/// - Every handle and binding handed to a function came from this pool's
///   [`try_reserve`](Self::try_reserve) or [`alloc`](Self::alloc), or was
///   bound to such a reservation by [`bind`](Self::bind).
/// - `failures` is the [`ErrorGraph`] that holds the failures this pool's
///   slices carry.
///
/// # Guarantees
///
/// - A slice is handed out again only once every owner of its last
///   allocation let go of it.
/// - A slice that stops existing releases the failure it carried on the spot.
pub trait SliceReserver {
    /// Whether the pool serves an allocation of `size` bytes.
    fn accepts_bytes(&self, size: u64) -> bool;

    /// Bind `assigned`, a handle no reservation placed yet, to the slice
    /// `reserved` holds, at `cursor`: the point of the stream from which the
    /// binding is valid. The claims the slice's last allocation carried are
    /// released.
    ///
    /// # Errors
    ///
    /// An implementation may answer [`IoError::NotFound`] when it can tell
    /// `reserved` names no slice it holds.
    ///
    /// # Panics
    ///
    /// An implementation may panic when `reserved` did not come from it.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError>;

    /// The slice `binding`'s location names.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when the location names no slice the pool holds.
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError>;

    /// The slice `binding`'s location names, mutably: the path the taint
    /// bookkeeping takes to reach the slice's claims.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when the location names no slice the pool holds.
    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError>;

    /// Reserve `size` bytes in the room the pool holds, `None` when that room
    /// has none for them; [`alloc`](Self::alloc) adds room.
    ///
    /// Reserving is one of the paths where a pool sheds slices (a paged pool
    /// coalesces on every attempt), so `failures` takes the release of the
    /// failure a shed slice carried.
    fn try_reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle>;

    /// Add room for `size` bytes and reserve them on it.
    ///
    /// `mapping` asks for real device backing now (`Eager`) or on first
    /// resolution (`Lazy`). A pool without lazy support may treat `Lazy` as
    /// `Eager`: the request is an opportunity to allocate less, never a
    /// promise the caller may rely on.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses, and [`IoError::BufferTooBig`] from an
    /// implementation that answers a size [`accept`](Self::accepts_bytes) refuses.
    ///
    /// # Panics
    ///
    /// An implementation that carves pages may panic when `size` is one
    /// [`accept`](Self::accepts_bytes) refuses.
    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError>
    where
        Self: Sized;

    /// Reserve `size` bytes in the room the pool holds, and add room for them
    /// when it holds none.
    ///
    /// # Errors
    ///
    /// As [`alloc`](Self::alloc).
    ///
    /// # Panics
    ///
    /// As [`alloc`](Self::alloc).
    fn reserve<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError>
    where
        Self: Sized,
    {
        match self.try_reserve(size, failures) {
            Some(handle) => Ok(handle),
            None => self.alloc(storage, size, mapping, failures),
        }
    }

    /// Give the allocation behind `binding` real device backing, if it was
    /// made [`PageMapping::Lazy`], before its storage handle reaches
    /// [`StorageAllocator::get`]. A pool that only allocates eagerly does
    /// nothing, and so does every pool for a binding it does not hold: lookup
    /// errors are [`find`](Self::find)'s to report.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    fn materialize<Storage: StorageAllocator>(
        &mut self,
        _storage: &mut Storage,
        _binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// The pool's shape, usage and high-water marks.
    fn report(&self) -> MemoryPoolReport;

    /// Keep the page `location` names as it is for as long as the guard lives
    /// (see [`PageGuard`]). `None` when the pool holds no such page.
    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard>;

    /// Give back what the pool holds and nothing uses, as far as `cleanup`
    /// asks, `reservation_count` reservations into the memory's life.
    fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) where
        Self: Sized;

    /// A storage handle for `size` bytes honoring `mapping`: a real device
    /// allocation when [`Eager`](PageMapping::Eager), a minted id awaiting
    /// [`materialize`](Self::materialize) when [`Lazy`](PageMapping::Lazy).
    /// Every implementation allocates its storage through it.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses, for an eager mapping.
    fn page_storage<Storage: StorageAllocator>(
        mapping: PageMapping,
        storage: &mut Storage,
        size: u64,
    ) -> Result<StorageHandle, IoError>
    where
        Self: Sized,
    {
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
    /// says: what an implementation whose slices own their buffer allocates.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses.
    fn whole_slice<Storage: StorageAllocator>(
        storage: &mut Storage,
        size: u64,
        padded_size: u64,
        mapping: PageMapping,
        location: MemoryLocation,
    ) -> Result<Slice, IoError>
    where
        Self: Sized,
    {
        let mut slice = Slice::new(Self::page_storage(mapping, storage, padded_size)?, 0);
        slice.mapped = mapping == PageMapping::Eager;
        fit_whole_slice(&mut slice, size, padded_size);
        slice.descriptor().update_location(location);
        Ok(slice)
    }

    /// Give a slice that owns its whole allocation, found for `binding`, real
    /// device backing if it was allocated [`PageMapping::Lazy`]: allocate for
    /// real and retire the minted id, which never reached the driver. What an
    /// implementation whose slices own their buffer materializes with.
    ///
    /// A missing slice, a mapped one and one `binding` has no claim on are left
    /// alone: lookup errors are the lookup's to report, and backing a slice
    /// nobody asked to resolve would allocate device memory for nothing.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the allocation.
    fn materialize_whole<Storage: StorageAllocator>(
        slice: Option<&mut Slice>,
        binding: &ManagedMemoryBinding,
        storage: &mut Storage,
    ) -> Result<(), IoError>
    where
        Self: Sized,
    {
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
    fn release_slice<Storage: StorageAllocator>(
        slice: &mut Slice,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) where
        Self: Sized,
    {
        slice.tainted.clear(failures);
        Self::deallocate_mapped(slice.mapped, slice.storage.id, storage);
    }

    /// Return the allocation `id` to the driver, when it is `mapped`.
    fn deallocate_mapped<Storage: StorageAllocator>(
        mapped: bool,
        id: StorageId,
        storage: &mut Storage,
    ) where
        Self: Sized,
    {
        if mapped {
            storage.deallocate(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::memory_pool::DirectSliceReserver;
    use crate::storage::BytesStorage;

    type Pool = DirectSliceReserver;

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

    /// A lazy page mints an id the storage never allocated, and an eager one
    /// is allocated now.
    #[test]
    fn only_an_eager_page_is_allocated_now() {
        let mut storage = BytesStorage::default();

        let lazy = Pool::page_storage(PageMapping::Lazy, &mut storage, 64)
            .expect("a minted id should never fail");
        assert_eq!((lazy.size(), storage.bytes_allocated()), (64, 0));

        Pool::page_storage(PageMapping::Eager, &mut storage, 64)
            .expect("the host should have 64 bytes");
        assert_eq!(storage.bytes_allocated(), 64);
    }

    /// A whole slice serves its size at its location, padded to the rest.
    #[test]
    fn a_whole_slice_serves_its_size_at_its_location() {
        let mut storage = BytesStorage::default();

        let whole = Pool::whole_slice(
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

    /// Backing is installed only for an unmapped slice the binding owns.
    #[test]
    fn only_an_unmapped_slice_the_binding_owns_is_backed() {
        let mut storage = BytesStorage::default();
        let mut owned = slice(64);
        owned.mapped = false;
        let binding = owned.handle.clone().binding();

        Pool::materialize_whole(None, &binding, &mut storage).expect("nothing should be backed");
        Pool::materialize_whole(
            Some(&mut owned),
            &ManagedMemoryHandle::new().binding(),
            &mut storage,
        )
        .expect("a foreign binding should be left alone");
        assert!(!owned.mapped);

        Pool::materialize_whole(Some(&mut owned), &binding, &mut storage)
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

        Pool::release_slice(&mut minted, &mut storage, &mut failures);
        Pool::release_slice(&mut backed, &mut storage, &mut failures);
        storage.flush();

        assert_eq!(storage.bytes_allocated(), 0);
    }
}
