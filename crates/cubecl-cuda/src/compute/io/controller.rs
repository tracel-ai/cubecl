use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController {
    /// The memory binding, kept alive until deallocation.
    binding: Option<SliceBinding>,
}

impl PinnedMemoryManagedAllocController {
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
    pub fn init(binding: SliceBinding, resource: PinnedMemoryResource) -> (Self, Allocation) {
        let ptr = resource.ptr;
        let size = resource.size;

        let allocation = Allocation {
            ptr: NonNull::new(ptr).expect("Pinned memory pointer must not be null"),
            size,
            align: PINNED_MEMORY_ALIGNMENT,
        };

        (
            Self {
                binding: Some(binding),
            },
            allocation,
        )
    }
}

impl AllocationController for PinnedMemoryManagedAllocController {
    fn dealloc(&mut self, _allocation: &Allocation) {
        self.binding = None;
    }
}
