use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use cubecl_common::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty};
use cubecl_runtime::memory_management::ManagedMemoryBinding;

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController {
    resource: PinnedMemoryResource,
    /// The memory binding, kept alive until deallocation.
    _binding: ManagedMemoryBinding,
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
    pub fn init(binding: ManagedMemoryBinding, resource: PinnedMemoryResource) -> Self {
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

    // Pinned host memory is always host-resident: the policy never forces a copy here.
    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [std::mem::MaybeUninit<u8>], AccessError> {
        // A zero-size resource carries a NULL pointer (a zero-size host alloc
        // returns success without allocating), which `from_raw_parts_mut`
        // rejects even for an empty slice — hand out an aligned dangling
        // pointer instead.
        if self.resource.size == 0 {
            return Ok(empty_pinned_slice_mut());
        }
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        // - Caller has to promise to only write initialized data to this slice.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(
                self.resource.ptr as *mut std::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[std::mem::MaybeUninit<u8>], AccessError> {
        // See `memory_mut`: a zero-size resource carries a NULL pointer.
        if self.resource.size == 0 {
            return Ok(empty_pinned_slice_mut());
        }
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.resource.ptr as *mut std::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        })
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }
}

/// An empty slice whose (dangling) pointer still satisfies
/// [`PINNED_MEMORY_ALIGNMENT`], matching what `alloc_align` advertises.
fn empty_pinned_slice_mut<'a>() -> &'a mut [std::mem::MaybeUninit<u8>] {
    // SAFETY: a dangling, well-aligned, non-null pointer is valid for a
    // zero-length slice.
    unsafe {
        std::slice::from_raw_parts_mut(std::ptr::without_provenance_mut(PINNED_MEMORY_ALIGNMENT), 0)
    }
}
