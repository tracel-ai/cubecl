use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_core::server::IoError;
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

use crate::compute::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};

pub struct PinnedMemoryManagedAllocController {
    binding: Option<SliceBinding>,
}

impl AllocationController for PinnedMemoryManagedAllocController {
    fn dealloc(&mut self, _allocation: &Allocation) {
        self.binding = None;
    }
}

impl PinnedMemoryManagedAllocController {
    pub fn init(
        binding: SliceBinding,
        resource: PinnedMemoryResource,
    ) -> Result<(Self, Allocation), IoError> {
        let ptr = resource.ptr;
        let size = resource.size;

        let allocation = Allocation {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            align: PINNED_MEMORY_ALIGNMENT,
        };

        Ok((
            Self {
                binding: Some(binding),
            },
            allocation,
        ))
    }
}
