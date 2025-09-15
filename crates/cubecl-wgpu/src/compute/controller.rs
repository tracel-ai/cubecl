use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    staging_buffer: wgpu::Buffer,
    binding: Option<SliceBinding>,
}

impl AllocationController for WgpuAllocController {
    fn dealloc(&mut self, _allocation: &cubecl_common::bytes::Allocation) {
        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.staging_buffer.unmap();
        self.binding = None;
    }
}

impl WgpuAllocController {
    /// Creates a new allocation controller for a managed wgpu staging buffer.
    ///
    /// # Arguments
    ///
    /// * `binding` - The memory binding for the managed buffer.
    /// * `buffer` - The wgpu buffer.
    /// * `size` - The size of the buffer in used for the copy.
    ///
    /// # Returns
    ///
    /// The controller and the corresponding `Allocation`.
    pub fn init(binding: SliceBinding, buffer: wgpu::Buffer, size: usize) -> (Self, Allocation) {
        fn buffer_ptr(buffer: &wgpu::Buffer) -> NonNull<u8> {
            let data = buffer.slice(..).get_mapped_range();
            unsafe { NonNull::new_unchecked(data.as_ptr().cast_mut()) }
        }

        let allocation = Allocation {
            ptr: buffer_ptr(&buffer),
            size,
            align: wgpu::COPY_BUFFER_ALIGNMENT as usize,
        };

        (
            Self {
                staging_buffer: buffer,
                binding: Some(binding),
            },
            allocation,
        )
    }
}
