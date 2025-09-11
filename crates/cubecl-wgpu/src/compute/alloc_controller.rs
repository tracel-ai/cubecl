use std::ptr::NonNull;

use cubecl_common::bytes::{Allocation, AllocationController};

pub struct WgpuAllocController {
    staging_buffer: wgpu::Buffer,
}

impl AllocationController for WgpuAllocController {
    fn dealloc(&self, _allocation: &cubecl_common::bytes::Allocation) {
        self.staging_buffer.unmap();
        self.staging_buffer.destroy();
    }

    fn grow(
        &self,
        _allocation: &cubecl_common::bytes::Allocation,
        _size: usize,
        _align: usize,
    ) -> Result<cubecl_common::bytes::Allocation, cubecl_common::bytes::AllocationError> {
        Err(cubecl_common::bytes::AllocationError::UnsupportedOperation)
    }

    fn can_be_detached(&self) -> bool {
        false
    }
}

impl WgpuAllocController {
    pub fn init(buffer: wgpu::Buffer, size: usize, align: usize) -> (Self, Allocation) {
        fn buffer_ptr(buffer: &wgpu::Buffer) -> NonNull<u8> {
            let data = buffer.slice(..).get_mapped_range();
            unsafe { NonNull::new_unchecked(data.as_ptr().cast_mut()) }
        }

        let allocation = Allocation {
            ptr: buffer_ptr(&buffer),
            size,
            align,
        };

        (
            Self {
                staging_buffer: buffer,
            },
            allocation,
        )
    }
}
