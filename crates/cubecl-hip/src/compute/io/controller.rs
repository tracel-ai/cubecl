use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController<'a> {
    allocation: Allocation<'a>,

    /// The memory binding, kept alive until deallocation.
    _binding: SliceBinding,
}

impl PinnedMemoryManagedAllocController<'_> {
    /// Creates a new allocation controller for pinned host memory.
    ///
    /// # Arguments
    ///
    /// * `binding` - The memory binding for the pinned memory.
    /// * `resource` - The pinned memory resource to manage.
    ///
    /// # Returns
    ///
    /// The controller and the corresponding `Allocation`.
    ///
    /// # Panics
    ///
    /// Panics if the provided `resource.ptr` is a null pointer.
    pub fn init(binding: SliceBinding, resource: PinnedMemoryResource) -> Self {
        // SAFETY:
        // - The ptr is valid for 'a as own the binding for the 'a lifetime.
        // - The allocation is from a resource of the size `size` and
        //   alignment `PINNED_MEMORY_ALIGNMENT`.
        let allocation = unsafe {
            Allocation::new_init(
                NonNull::new(resource.ptr).expect("Pinned memory pointer must not be null"),
                resource.size,
                PINNED_MEMORY_ALIGNMENT,
            )
        };

        Self {
            _binding: binding,
            allocation,
        }
    }
}

impl AllocationController for PinnedMemoryManagedAllocController<'_> {
    fn alloc_align(&self) -> usize {
        self.allocation.align()
    }

    fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        self.allocation.memory_mut()
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        self.allocation.memory()
    }
}
