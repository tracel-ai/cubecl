use cubecl_common::bytes::{Allocation, AllocationController};
use cubecl_runtime::memory_management::SliceBinding;
use std::ptr::NonNull;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    staging_buffer: wgpu::Buffer,
    bindings: Option<(SliceBinding, wgpu::BufferViewMut<'static>)>,
}

impl AllocationController for WgpuAllocController {
    fn dealloc(&mut self, _allocation: &cubecl_common::bytes::Allocation) {
        let old = self.bindings.take();
        core::mem::drop(old);
        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.staging_buffer.unmap();
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
    pub fn init(
        binding: SliceBinding,
        mut buffer: wgpu::Buffer,
        size: usize,
    ) -> (Self, Allocation) {
        fn buffer_ptr(buffer: &mut wgpu::Buffer) -> (NonNull<u8>, wgpu::BufferViewMut<'static>) {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            let data: &mut [u8] = view.as_mut();

            unsafe {
                (NonNull::new_unchecked(data.as_mut_ptr()), {
                    core::mem::transmute(view)
                })
            }
        }

        let (ptr, view) = buffer_ptr(&mut buffer);
        let allocation = Allocation {
            ptr,
            size,
            align: wgpu::COPY_BUFFER_ALIGNMENT as usize,
        };

        (
            Self {
                staging_buffer: buffer,
                bindings: Some((binding, view)),
            },
            allocation,
        )
    }
}
