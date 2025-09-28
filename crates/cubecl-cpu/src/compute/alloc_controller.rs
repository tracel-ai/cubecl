use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_core::server::{Binding, IoError};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::{alloc::Layout, ptr::NonNull};

pub struct CpuAllocController<'a> {
    allocation: Allocation<'a>,
    // Needed to keep the binding alive.
    _binding: Option<Binding>,
}

impl AllocationController for CpuAllocController<'_> {
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

impl CpuAllocController<'_> {
    pub fn init(
        binding: Binding,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Result<Self, IoError> {
        let resource = memory_management
            .get_resource(
                binding.memory.clone(),
                binding.offset_start,
                binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)?;

        let write = resource.write();
        let layout = Layout::for_value(write);
        let ptr =
            NonNull::new(resource.write().as_mut_ptr()).expect("Resource pointers cannot be null");

        // SAFETY:
        // - The ptr is valid and points to a memory region allocated by the system.
        // - The size and alignment are correct for the layout.
        let allocation = unsafe { Allocation::new_init(ptr, layout.size(), layout.align()) };

        Ok(Self {
            _binding: Some(binding),
            allocation,
        })
    }
}
