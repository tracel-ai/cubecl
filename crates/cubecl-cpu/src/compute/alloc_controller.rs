use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, Reader, Writer,
};
use cubecl_core::server::IoError;
use cubecl_runtime::{
    memory_management::{ManagedMemoryBinding, MemoryManagement},
    storage::{BytesResource, BytesStorage},
};

pub struct CpuAllocController {
    resource: BytesResource,
    // Needed to keep the binding alive.
    _binding: ManagedMemoryBinding,
    /// Private copy of the data, created on the first mutable access (copy-on-write).
    private: Option<Bytes>,
}

impl AllocationController for CpuAllocController {
    fn alloc_align(&self) -> usize {
        match &self.private {
            Some(bytes) => bytes.align(),
            None => align_of::<u8>(),
        }
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }

    /// SAFETY:
    /// - The caller must ensure only initialized memory is written.
    // Reads are zero-copy: the returned memory aliases the live pool allocation, which the
    // server keeps owning (and which live tensors may still reference). Writing through that
    // alias would corrupt the allocation, so mutation copies into a private buffer first.
    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [std::mem::MaybeUninit<u8>], AccessError> {
        if self.private.is_none() {
            if !policy.copy_allowed() {
                return Err(AccessError::WouldCopy);
            }
            self.private = Some(Bytes::from_bytes_vec(self.resource.read().to_vec()));
        }

        let slice = self
            .private
            .as_mut()
            .expect("private buffer set above")
            .write(Writer::new())?;

        // SAFETY:
        // - MaybeUninit has the same layout as u8.
        // - Caller upholds only writing initialized memory.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                slice.len(),
            )
        })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[std::mem::MaybeUninit<u8>], AccessError> {
        let slice = match &self.private {
            // Once copy-on-write happened, the private buffer is the source of truth.
            Some(bytes) => bytes.read(Reader::new())?,
            None => self.resource.read(),
        };

        // SAFETY:
        // - MaybeUninit has the same layout as u8.
        Ok(unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const std::mem::MaybeUninit<u8>,
                slice.len(),
            )
        })
    }
}

impl CpuAllocController {
    pub fn init(
        binding: cubecl_core::server::BufferBinding,
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
            private: None,
        })
    }
}
