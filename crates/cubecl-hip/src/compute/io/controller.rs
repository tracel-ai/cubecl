use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use cubecl_common::bytes::{AllocationController, AllocationProperty};
use cubecl_runtime::memory_management::SliceBinding;

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController {
    resource: PinnedMemoryResource,

    /// The memory binding, kept alive until deallocation.
    _binding: SliceBinding,
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
    pub fn init(binding: SliceBinding, resource: PinnedMemoryResource) -> Self {
        Self {
            _binding: binding,
            resource,
        }
    }
}

impl AllocationController for PinnedMemoryManagedAllocController {
    fn alloc_align(&self) -> usize {
        PINNED_MEMORY_ALIGNMENT
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    unsafe fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        // - Caller has to promise to only write initialized data to this slice.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.resource.ptr as *mut std::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        unsafe {
            std::slice::from_raw_parts(
                self.resource.ptr as *mut std::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        }
    }
}
