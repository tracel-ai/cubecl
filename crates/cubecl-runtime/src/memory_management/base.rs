use core::fmt::Debug;
use crate::storage::{ComputeStorage, StorageHandle};
use super::MemoryLock;

/// Amount of memory in use by this allocator
/// and statistics on how much memory is reserved and
/// wasted in total.
pub struct MemoryUsage {
    /// The number of allocations currently active.
    pub number_allocs: usize,
    /// The number of bytes that are currently actually in use.
    ///
    /// This doesn't include any padding or other memory that needs to be
    /// reserved, and is the minimum amount of memory that could possible
    /// be allocated.
    pub bytes_in_use: usize,
    /// The amount of bytes used for padding memory in currently active allocations.
    pub bytes_padding: usize,
    /// The total amount of memory reserved on the device.
    ///
    /// This will be at least as much as bytes_in_use but in practice will
    /// be higher, as allocations reserve memory for future allocations
    /// and for padding.
    pub bytes_reserved: usize,
}

impl MemoryUsage {
    pub(crate) fn combine(&self, other: MemoryUsage) -> MemoryUsage {
        MemoryUsage {
            number_allocs: self.number_allocs + other.number_allocs,
            bytes_in_use: self.bytes_in_use + other.bytes_in_use,
            bytes_padding: self.bytes_padding + other.bytes_padding,
            bytes_reserved: self.bytes_reserved + other.bytes_reserved,
        }
    }
}

impl std::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // In the future it'd be nice if MemoryUsage also held some stats about say,
        // the 5 biggest allocations, to show when you an OOM.
        let usage_percentage = (self.bytes_in_use as f32 / self.bytes_reserved as f32) * 100.0;
        let padding_percentage = (self.bytes_padding as f32 / self.bytes_in_use as f32) * 100.0;
        writeln!(f, "Memory Usage Report:")?;
        writeln!(f, "  Number of allocations: {}", self.number_allocs)?;
        writeln!(f, "  Bytes in use: {} bytes", self.bytes_in_use)?;
        writeln!(f, "  Bytes used for padding: {} bytes", self.bytes_padding)?;
        writeln!(f, "  Total bytes reserved: {} bytes", self.bytes_reserved)?;
        writeln!(f, "  Usage efficiency: {:.2}%", usage_percentage)?;
        writeln!(f, "  Padding overhead: {:.2}%", padding_percentage)
    }
}

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
    fn reserve(&mut self, size: usize, locked: Option<&MemoryLock>) -> Self::Handle;

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

    /// Get the current memory usage.
    fn memory_usage(&self) -> MemoryUsage;
}
