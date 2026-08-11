use super::{ManagedMemoryBinding, ManagedMemoryDescriptor, ManagedMemoryHandle};
use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

/// Whether a fresh device page gets real backing at allocation time.
///
/// Either way the pool's bookkeeping — slice carving, coalescing, high-water
/// marks — is identical; the two differ only in when the driver is asked for
/// memory. [`page_mapping`] decides which an allocation gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageMapping {
    /// Allocate device memory now.
    Eager,
    /// Mint the storage id now and defer the device allocation to the first
    /// time one of the page's slices is resolved
    /// ([`MemoryPool::materialize`]). Until then the id has no driver memory
    /// behind it and the page's device footprint is zero.
    Lazy,
}

/// What backing an allocation made on this thread, right now, should get.
///
/// **[`Eager`](PageMapping::Eager) is the answer whenever the allocation will
/// be used**, which outside a dry run is all of them: a reservation becomes a
/// kernel argument, a read or a write within microseconds, so deferring it
/// buys nothing and gives up three things.
///
/// - **A failure that can still be recovered.** A device allocation that fails
///   at reservation is retried by the backends' `Command::reserve` after
///   reclaiming the stream — the transient-peak case, a build holding float
///   weights while their quantized copies allocate. Deferred, the same failure
///   lands at resolution, where there is nothing left to flush and no retry.
/// - **Infallible resolution.** Resolving a binding is a pure lookup today,
///   and several launch paths resolve with `expect`. Allocating there would
///   turn a full device into a panic instead of a queued error.
/// - **A capture window that cannot allocate.** Graph capture needs every
///   slice the recorded run touches to exist before recording starts. Eager
///   reservation is what guarantees that; deferred, a slice warmup reserved
///   but never resolved would ask the driver for memory mid-capture and fault.
///
/// **[`Lazy`](PageMapping::Lazy) is for the one case where the allocation may
/// never be used**: under a [`DryRun`](crate::dry_run::DryRun) the workload's
/// launches are compiled and dropped, so most reservations are never resolved
/// and never need to exist. That is what lets a workload far larger than the
/// device replay its allocation stream — the pools still measure it, and only
/// what genuinely executes (a tuning pass, which resolves because it runs)
/// costs real memory.
pub fn page_mapping() -> PageMapping {
    match crate::dry_run::dry_run() {
        true => PageMapping::Lazy,
        false => PageMapping::Eager,
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
    /// Whether `storage.id` is backed by a real device allocation. Only pools
    /// whose slices own their whole buffer (the persistent pool) track
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
