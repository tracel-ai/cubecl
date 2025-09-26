use cubecl_common::bytes::{Allocation, BytesBacking};
use cubecl_core::server::{Binding, IoError};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

pub struct CpuAllocController<'a> {
    binding: Option<Binding>,
    allocation: Allocation<'a>,
}

impl BytesBacking for CpuAllocController<'_> {
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
        let size = layout.size();
        let align = layout.align();
        let ptr = resource.write().as_mut_ptr();

        let allocation = Allocation {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            align,
            _lifetime: PhantomData,
        };

        Ok(Self {
            binding: Some(binding),
            allocation,
        })
    }
}
