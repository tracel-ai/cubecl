use super::{ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice};
use crate::{
    memory_management::{Cleanup, ErrorGraph, MemoryLocation, PageGuard},
    server::IoError,
    storage::StorageAllocator,
};

/// Reserves slices of device memory for allocations, and reuses them once
/// their owners let go.
///
/// The functions that take a storage are for callers holding the pool's own
/// type; the rest can be reached through `dyn SliceReserver`, which is how a
/// location is resolved to whichever pool it names.
///
/// # Assumptions
///
/// `failures` is the [`ErrorGraph`] that holds the failures this pool's
/// slices carry.
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
    /// # Assumptions
    ///
    /// `reserved` came from this pool's [`try_reserve`](Self::try_reserve) or
    /// [`alloc`](Self::alloc).
    ///
    /// # Panics
    ///
    /// When `reserved` names no slice the pool holds.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    );

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
    /// # Assumptions
    ///
    /// `size` is one [`accepts_bytes`](Self::accepts_bytes) takes.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses.
    ///
    /// # Panics
    ///
    /// An implementation may panic on a size
    /// [`accepts_bytes`](Self::accepts_bytes) refuses.
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
    /// # Assumptions
    ///
    /// `size` is one [`accepts_bytes`](Self::accepts_bytes) takes.
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
}
