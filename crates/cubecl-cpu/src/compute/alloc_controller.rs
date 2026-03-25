use cubecl_common::bytes::{AllocationController, AllocationProperty};
use cubecl_core::server::IoError;
use cubecl_runtime::{
    memory_management::{ManagedMemoryBinding, MemoryManagement},
    storage::{BytesResource, BytesStorage},
};

pub struct CpuAllocController {
    resource: BytesResource,
    // Needed to keep the binding alive.
    _binding: ManagedMemoryBinding,
}

impl AllocationController for CpuAllocController {
    fn alloc_align(&self) -> usize {
        align_of::<u8>()
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }

    /// SAFETY:
    /// - The caller must ensure only initialized memory is written.
    unsafe fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        let slice = self.resource.write();

        // SAFETY:
        // - MaybeUninit has the same layout as u8.
        // - Caller upholds only writing initialized memory.
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                slice.len(),
            )
        }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        let slice = self.resource.read();

        // SAFETY:
        // - MaybeUninit has the same layout as u8.
        unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const std::mem::MaybeUninit<u8>,
                slice.len(),
            )
        }
    }
}

impl CpuAllocController {
    pub fn init(
        binding: cubecl_core::server::Binding,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Result<Self, IoError> {
        let memory = binding.memory.clone();
        let resource = memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )?;

        Ok(Self {
            _binding: memory,
            resource,
        })
    }
}
