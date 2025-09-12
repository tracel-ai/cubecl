use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_core::server::{Binding, IoError};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::{alloc::Layout, ptr::NonNull};

pub struct CpuAllocController {
    binding: Option<Binding>,
}

impl AllocationController for CpuAllocController {
    fn dealloc(&mut self, _allocation: &Allocation) {
        self.binding = None;
    }
}

impl CpuAllocController {
    pub fn init(
        binding: Binding,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Result<(Self, Allocation), IoError> {
        let resource = memory_management
            .get_resource(
                binding.memory.clone(),
                binding.offset_start,
                binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)?;

        let write = resource.write();
        let layout = Layout::for_value(write);
        let size = layout.size();
        let align = layout.align();
        let ptr = resource.write().as_mut_ptr();

        let allocation = Allocation {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            align,
        };

        Ok((
            Self {
                binding: Some(binding),
            },
            allocation,
        ))
    }
}
