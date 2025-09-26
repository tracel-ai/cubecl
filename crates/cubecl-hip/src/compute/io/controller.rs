use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use cubecl_common::bytes::{Allocation, BytesBacking};
use cubecl_runtime::memory_management::SliceBinding;
use std::{marker::PhantomData, ptr::NonNull};

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController<'a> {
    /// The memory binding, kept alive until deallocation.
    binding: Option<SliceBinding>,

    allocation: Allocation<'a>,
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
        let ptr = resource.ptr;
        let size = resource.size;

        let allocation = Allocation {
            ptr: NonNull::new(ptr).expect("Pinned memory pointer must not be null"),
            size,
            align: PINNED_MEMORY_ALIGNMENT,
            _lifetime: PhantomData,
        };

        Self {
            allocation,
            binding: Some(binding),
        }
    }
}

impl BytesBacking for PinnedMemoryManagedAllocController<'_> {
    fn dealloc(&mut self) {
        self.binding = None;
    }

    fn alloc_align(&self) -> usize {
        self.allocation.align
    }

    fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        self.allocation.memory_mut()
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        self.allocation.memory()
    }
}
