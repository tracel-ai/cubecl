use cubecl_common::{
    backtrace::BackTrace,
    bytes::{AllocationController, AllocationProperty},
};
use cubecl_core::server::IoError;
use cubecl_runtime::{
    memory_management::{ManagedMemoryHandle, MemoryManagement},
    storage::{BytesResource, BytesStorage},
};

pub struct CpuAllocController {
    resource: BytesResource,
    // Needed to keep the binding alive.
    _handle: ManagedMemoryHandle,
}

impl AllocationController for CpuAllocController {
    fn alloc_align(&self) -> usize {
        align_of::<u8>()
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }

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
        handle: cubecl_core::server::Binding,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Result<Self, IoError> {
        let slot = memory_management.get_slot(handle)?;
        let handle = slot.memory.clone();

        let resource = memory_management
            .get_resource(slot.memory.binding(), slot.offset_start, slot.offset_end)
            .ok_or(IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })?;

        Ok(Self {
            _handle: handle,
            resource,
        })
    }
}
