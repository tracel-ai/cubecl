use core::mem::MaybeUninit;
use core::pin::Pin;
use cubecl_common::bytes::{AllocationController, AllocationProperty};
use cubecl_runtime::memory_management::SliceBinding;
use wgpu::{BufferSlice, BufferViewMut};

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController<'a> {
    view: Option<BufferViewMut<'a>>,
    buffer: Pin<Box<wgpu::Buffer>>,
    _binding: SliceBinding,
}

impl Drop for WgpuAllocController<'_> {
    fn drop(&mut self) {
        // Drop the view first, then unmap the buffer.
        // This ensures proper cleanup order since the view borrows from the buffer.
        drop(self.view.take());
        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.buffer.unmap();
    }
}

impl AllocationController for WgpuAllocController<'_> {
    fn alloc_align(&self) -> usize {
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
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

impl<'a> WgpuAllocController<'a> {
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
        let buf = Box::pin(buffer);
        let slice = buf.slice(..);

        // SAFETY: We're extending the lifetime to match the controller's lifetime. Internally the BufferViewMut holds
        // a reference to the buffer.
        //
        // - The view is always dropped before the buffer
        // - The buffer stays alive as long as the controller exists
        // - The buffer is pinned and will never move after creating the view
        let slice = unsafe { std::mem::transmute::<BufferSlice<'_>, BufferSlice<'a>>(slice) };

        Self {
            view: Some(slice.get_mapped_range_mut()),
            buffer: buf,
            _binding: binding,
        }
    }
}
