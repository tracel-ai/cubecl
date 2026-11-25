use super::{ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice};
use crate::{
    memory_management::{Cleanup, ErrorGraph, MemoryLocation, MemoryUsage, PageGuard},
    server::IoError,
    storage::StorageAllocator,
};

/// Declares how memory is allocated in a reusable pool.
///
/// The methods that take a storage are for callers holding the pool's own
/// type; the rest can be reached through `dyn MemoryPool`, which is how a
/// location is resolved to whichever pool it names.
pub trait MemoryPool {
    /// Whether the memory pool accepts the given size.
    fn accept(&self, size: u64) -> bool;

    /// Binds an uninitialized handle to a previously reserved memory slice.
    ///
    /// # Arguments
    ///
    /// * `reserved` - An existing, initialized handle representing the underlying memory allocation.
    /// * `assigned` - A new handle that will be initialized to point into the `reserved` memory region.
    /// * `cursor` - A sequence point or timestamp determining when this binding becomes valid for access.
    ///
    /// # Errors
    ///
    /// Returns [`IoError`] if the reservation is invalid or if the cursor position
    /// is outside the bounds of the memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError>;

    /// Retrieves the slice for the binding.
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError>;

    /// Retrieves the slice for the binding, mutably — the path the taint
    /// bookkeeping takes to reach [`Slice::tainted`].
    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError>;

    /// Try to reserve a memory slice of the given size.
    ///
    /// `failures` is threaded through because reserving is one of the paths
    /// where a pool sheds slices (a sliced pool coalesces on every attempt),
    /// and a shed slice must release the failure it carried on the spot.
    ///
    /// # Notes
    ///
    /// It is not guaranteed the `try_reserve` function will reapply the accept function.
    /// Therefore it is a good idea to call [`MemoryUsage::accept()`] before using `try_reserve`.
    ///
    /// # Returns
    ///
    /// A [slice handle](StorageHandle) if the current memory pool has enough memory, otherwise it
    /// will returns [None]. You can then call [`MemoryPool::alloc()`] to increase the amount of
    /// memory the pool has.
    fn try_reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle>;

    /// Increases the amount of memory the pool has and returns a [slice handle](StorageHandle)
    /// corresponding to the requested size.
    ///
    /// # Notes
    ///
    /// The function uses a [`StorageAllocator`] to perform the allocation. It might return an error
    /// if the allocation fails or if the requested size is bigger than the memory pool is
    /// configured to handle.
    ///
    /// `mapping` asks for real device backing now (`Eager`) or on first
    /// resolution (`Lazy`). A pool without lazy support may treat `Lazy` as
    /// `Eager`: the request is an opportunity to allocate less, never a
    /// promise the caller may rely on.
    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError>
    where
        Self: Sized;

    /// Reserve `size` bytes in the room the pool holds, and allocate a page
    /// for them when it holds none.
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

    /// Ensure the allocation behind `binding` has real device backing,
    /// installing it now if the allocation was made [`PageMapping::Lazy`].
    /// Must be called before the binding's storage handle reaches
    /// [`StorageAllocator::get`]. A no-op for pools that only allocate eagerly
    /// and for bindings this pool does not hold (lookup errors surface from
    /// [`find`](Self::find), not from here).
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

    /// Computes the [`MemoryUsage`] for this pool.
    fn get_memory_usage(&self) -> MemoryUsage;

    /// Keep the page `location` names as it is for as long as the guard lives
    /// (see [`PageGuard`]). `None` when the pool holds no such page.
    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard>;

    /// Cleanup the memory pool, maybe freeing some memory using the [`StorageAllocator`].
    fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) where
        Self: Sized;
}
