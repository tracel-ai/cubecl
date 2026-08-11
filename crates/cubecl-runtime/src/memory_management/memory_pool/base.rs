use super::{ManagedMemoryBinding, ManagedMemoryDescriptor, ManagedMemoryHandle};
use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use cubecl_environment::backtrace::BackTrace;

/// Whether a fresh device page gets real backing at allocation time.
///
/// Either way the pool's bookkeeping — slice carving, coalescing, high-water
/// marks — is identical; the two differ only in when the driver is asked for
/// memory. [`PageMapping::current`] decides which an allocation gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageMapping {
    /// Allocate device memory now.
    ///
    /// The answer whenever the allocation will be used, which outside a dry
    /// run is all of them: a reservation becomes a kernel argument, a read or
    /// a write within microseconds, so deferring it buys nothing — and gives
    /// up the reservation-time failure the backends can still recover
    /// (`Command::reserve` retries after reclaiming the stream), the pure
    /// lookup that keeps resolution infallible on the launch paths, and the
    /// guarantee that a capture window never allocates once recording starts.
    Eager,
    /// Mint the storage id now and defer the device allocation to the first
    /// time one of the page's slices is resolved
    /// ([`MemoryPool::materialize`]). Until then the id has no driver memory
    /// behind it and the page's device footprint is zero.
    ///
    /// For the one case where the allocation may never be used: under a
    /// [`DryRun`](crate::dry_run::DryRun) the workload's launches are compiled
    /// and dropped, so most reservations are never resolved and never need to
    /// exist. That is what lets a workload far larger than the device replay
    /// its allocation stream — the pools still measure it, and only what
    /// genuinely executes (a tuning pass, which resolves because it runs)
    /// costs real memory.
    Lazy,
}

impl PageMapping {
    /// The mapping allocations made on this thread, right now, should get:
    /// [`Lazy`](Self::Lazy) under a [`DryRun`](crate::dry_run::DryRun),
    /// [`Eager`](Self::Eager) everywhere else.
    pub fn current() -> Self {
        match crate::dry_run::dry_run() {
            true => PageMapping::Lazy,
            false => PageMapping::Eager,
        }
    }

    /// A storage handle for `size` bytes honoring this mapping: a real device
    /// allocation when [`Eager`](Self::Eager), a minted id awaiting
    /// [`MemoryPool::materialize`] when [`Lazy`](Self::Lazy).
    pub(crate) fn storage_handle<Storage: ComputeStorage>(
        self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<StorageHandle, IoError> {
        match self {
            PageMapping::Eager => storage.alloc(size),
            PageMapping::Lazy => Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization { offset: 0, size },
            )),
        }
    }
}

/// Declares how memory is allocated in a reusable pool.
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
    ) -> Result<(), IoError>;

    /// Retrieves the slice for the binding.
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError>;

    /// Try to reserve a memory slice of the given size.
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
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle>;

    /// Increases the amount of memory the pool has and returns a [slice handle](StorageHandle)
    /// corresponding to the requested size.
    ///
    /// # Notes
    ///
    /// The function uses a [`ComputeStorage`] to perform the allocation. It might return an error
    /// if the allocation fails or if the requested size is bigger than the memory pool is
    /// configured to handle.
    ///
    /// `mapping` asks for real device backing now (`Eager`) or on first
    /// resolution (`Lazy`). A pool without lazy support may treat `Lazy` as
    /// `Eager`: the request is an opportunity to allocate less, never a
    /// promise the caller may rely on.
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
    ) -> Result<ManagedMemoryHandle, IoError>;

    /// Ensure the allocation behind `binding` has real device backing,
    /// installing it now if the allocation was made [`PageMapping::Lazy`].
    /// Must be called before the binding's storage handle reaches
    /// [`ComputeStorage::get`]. A no-op for pools that only allocate eagerly
    /// and for bindings this pool does not hold (lookup errors surface from
    /// [`find`](Self::find), not from here).
    fn materialize<Storage: ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        Ok(())
    }

    /// Computes the [`MemoryUsage`] for this pool.
    fn get_memory_usage(&self) -> MemoryUsage;

    /// Cleanup the memory pool, maybe freeing some memory using the [`ComputeStorage`].
    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    );
}

#[derive(Debug)]
/// Slice of data with its associated storage.
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: ManagedMemoryHandle,
    pub padding: u64,
    pub cursor: u64,
    /// Whether `storage.id` is backed by a real device allocation. Pools whose
    /// slices own their whole buffer (the persistent and direct pools) track
    /// laziness here; sliced pools track it on the page, whose id every slice
    /// shares.
    pub mapped: bool,
}

impl Slice {
    pub fn new(storage: StorageHandle, padding: u64) -> Self {
        Self {
            storage,
            handle: ManagedMemoryHandle::new(),
            padding,
            cursor: 0,
            mapped: true,
        }
    }
    /// If the slice is free to be reused.
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    /// The total size of the slice including padding.
    pub(crate) fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    /// The description of the slice.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        self.handle.descriptor()
    }

    /// Install real device backing behind a slice that owns its whole buffer
    /// and was allocated [`PageMapping::Lazy`]: allocate for real and retire
    /// the minted id, which never reached the driver. The caller checks the
    /// slice is unmapped and that the binding has a claim on it.
    pub(crate) fn materialize<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        let effective_size = self.effective_size();
        let real = storage
            .alloc(effective_size)
            .map_err(|err| IoError::StorageMappingFailed {
                size: effective_size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })?;
        self.storage.id = real.id;
        self.mapped = true;
        Ok(())
    }
}

/// Calculates the padding required to store the given size in a buffer given the memory alignment.
pub(crate) fn calculate_padding(size: u64, memory_alignment: u64) -> u64 {
    let remainder = size % memory_alignment;
    if remainder != 0 {
        memory_alignment - remainder
    } else {
        0
    }
}
