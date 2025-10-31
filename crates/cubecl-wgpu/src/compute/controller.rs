use core::mem::MaybeUninit;
use cubecl_common::bytes::AllocationController;
use cubecl_runtime::memory_management::SliceBinding;
use wgpu::BufferViewMut;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    view: Option<BufferViewMut>,
    buffer: wgpu::Buffer,
    _binding: SliceBinding,
}

impl Drop for WgpuAllocController {
    fn drop(&mut self) {
        // Drop the view first, then unmap the buffer.
        // This ensures proper cleanup order since the view borrows from the buffer.
        drop(self.view.take());
        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.buffer.unmap();
    }
}

impl AllocationController for WgpuAllocController {
    fn alloc_align(&self) -> usize {
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let bytes: &mut [u8] = self.view.as_mut().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        // - Caller promises not to write uninitialized values.
        unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut MaybeUninit<u8>, bytes.len())
        }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const MaybeUninit<u8>, bytes.len()) }
    }
}

impl WgpuAllocController {
    /// Creates a new allocation controller for a managed wgpu staging buffer.
    ///
    /// # Arguments
    ///
    /// * `binding` - The memory binding for the managed buffer.
    /// * `buffer` - The wgpu buffer.
    ///
    /// # Returns
    ///
    /// The controller.
    pub fn init(binding: SliceBinding, buffer: wgpu::Buffer) -> Self {
        let buf_view = buffer.slice(..).get_mapped_range_mut();

        Self {
            view: Some(buf_view),
            buffer,
            _binding: binding,
        }
    }
}
