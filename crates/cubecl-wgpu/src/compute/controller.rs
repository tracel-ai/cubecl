use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    binding: Option<SliceBinding>,
    buffer_data: Option<Box<[u8]>>,
}

impl AllocationController for WgpuAllocController {
    fn dealloc(&mut self, _allocation: &cubecl_common::bytes::Allocation) {
        self.buffer_data.take();
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
        let data = {
            let data = buffer.slice(..).get_mapped_range();
            data[0..size].to_vec().into_boxed_slice()
        };
        let ptr = unsafe { NonNull::new_unchecked(data.as_ptr().cast_mut()) };

        let allocation = Allocation {
            ptr,
            size,
            align: wgpu::COPY_BUFFER_ALIGNMENT as usize,
        };

        // We unmap the buffer and release the binding so that the same buffer can be used again.
        buffer.unmap();

        (
            Self {
                binding: Some(binding),
                buffer_data: Some(data),
            },
            allocation,
        )
    }
}
