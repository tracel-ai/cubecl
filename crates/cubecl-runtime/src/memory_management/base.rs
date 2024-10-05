use core::fmt::Debug;

use crate::storage::{ComputeStorage, StorageHandle, StorageId};

/// The managed tensor buffer handle that points to some memory segment.
/// It should not contain actual data.
pub trait MemoryHandle<Binding>: Clone + Send + Sync + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
    /// Get the binding associated to the current handle.
    fn binding(self) -> Binding;
}

/// The MemoryManagement trait encapsulates strategies for (de)allocating memory.
/// It is bound to the ComputeStorage trait, which does the actual (de)allocations.
///
/// The MemoryManagement can only reserve memory space or get the resource located at a space.
/// Modification of the resource data should be done directly on the resource.
pub trait MemoryManagement<Storage: ComputeStorage>: Send + core::fmt::Debug {
    /// The associated type that must implement [MemoryHandle].
    type Handle: MemoryHandle<Self::Binding>;
    /// The associated type that must implement [MemoryBinding]
    type Binding: Send + Sync + Clone + Debug;

    /// Returns the storage from the specified binding
    fn get(&mut self, binding: Self::Binding) -> StorageHandle;

    /// Returns the resource from the storage at the specified handle
    fn get_resource(
        &mut self,
        binding: Self::Binding,
        offset_start: Option<usize>,
        offset_end: Option<usize>,
    ) -> Storage::Resource {
        let handle = self.get(binding);
        let handle = match offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        self.storage().get(&handle)
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    fn reserve(&mut self, size: usize, exclude: &[StorageId]) -> Self::Handle;

    /// Bypass the memory allocation algorithm to allocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn alloc(&mut self, size: usize) -> Self::Handle;

    /// Bypass the memory allocation algorithm to deallocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn dealloc(&mut self, binding: Self::Binding);

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](MemoryManagement::alloc) and [dealloc](MemoryManagement::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    fn storage(&mut self) -> &mut Storage;
}
