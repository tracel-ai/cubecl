//! Page-locked host memory, and handing a slice of it out as [`Bytes`].
//!
//! Pinned pages are what a driver can DMA from without a bounce, so every
//! backend that stages transfers through the host allocates them. What one
//! looks like from the outside — a pointer, a length, and a binding that keeps
//! the allocation alive while a caller holds a slice of it — is the same
//! whichever driver page-locked it.

use crate::memory_management::ManagedMemoryBinding;
use cubecl_common::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty};

/// The alignment pinned allocations are handed out at.
///
/// A `u128`'s worth, which is the widest load a host-side copy will make of
/// staged bytes.
pub const PINNED_MEMORY_ALIGNMENT: usize = core::mem::size_of::<u128>();

/// A range of page-locked host memory.
#[derive(Debug)]
pub struct PinnedMemoryResource {
    /// Pointer to the pinned memory buffer.
    pub ptr: *mut u8,
    /// Size of the memory resource in bytes.
    pub size: usize,
}

// SAFETY: the pointer is to page-locked host memory, which stays valid and
// pinned whichever thread touches it; access is serialized by the device
// handle above it.
unsafe impl Send for PinnedMemoryResource {}

/// Hands out a pinned allocation as [`Bytes`](cubecl_common::bytes::Bytes),
/// keeping the allocation alive for as long as the bytes are.
///
/// The binding is held and never read: dropping it is what returns the pages
/// to the pool, so a caller that still has the slice still has the memory.
pub struct PinnedMemoryAllocController {
    resource: PinnedMemoryResource,
    /// The memory binding, kept alive until deallocation.
    _binding: ManagedMemoryBinding,
}

impl PinnedMemoryAllocController {
    /// A controller over the pinned allocation `binding` names, resolved to
    /// `resource`.
    pub fn init(binding: ManagedMemoryBinding, resource: PinnedMemoryResource) -> Self {
        Self {
            _binding: binding,
            resource,
        }
    }
}

impl AllocationController for PinnedMemoryAllocController {
    fn alloc_align(&self) -> usize {
        PINNED_MEMORY_ALIGNMENT
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    // Pinned host memory is always host-resident: the policy never forces a
    // copy here.
    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [core::mem::MaybeUninit<u8>], AccessError> {
        // A zero-size resource carries a NULL pointer — page-locking nothing
        // succeeds without allocating — which `from_raw_parts_mut` rejects
        // even for an empty slice. Hand out an aligned dangling pointer.
        if self.resource.size == 0 {
            return Ok(empty_pinned_slice_mut());
        }
        // SAFETY:
        // - the pointer is valid while the binding is alive,
        // - the resource was allocated with `size` bytes,
        // - `MaybeUninit<u8>` has the same layout as `u8`,
        // - the caller promises to write only initialized data into it.
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                self.resource.ptr as *mut core::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[core::mem::MaybeUninit<u8>], AccessError> {
        // See `memory_mut`: a zero-size resource carries a NULL pointer.
        if self.resource.size == 0 {
            return Ok(empty_pinned_slice_mut());
        }
        // SAFETY: as `memory_mut`, without the write.
        Ok(unsafe {
            core::slice::from_raw_parts(
                self.resource.ptr as *mut core::mem::MaybeUninit<u8>,
                self.resource.size,
            )
        })
    }
}

/// An empty slice whose dangling pointer still satisfies
/// [`PINNED_MEMORY_ALIGNMENT`], matching what `alloc_align` advertises.
fn empty_pinned_slice_mut<'a>() -> &'a mut [core::mem::MaybeUninit<u8>] {
    // SAFETY: a dangling, well-aligned, non-null pointer is valid for a
    // zero-length slice.
    unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::without_provenance_mut(PINNED_MEMORY_ALIGNMENT),
            0,
        )
    }
}
