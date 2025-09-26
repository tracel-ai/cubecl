use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use cubecl_common::bytes::BytesBacking;
use cubecl_runtime::memory_management::SliceBinding;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController<'a> {
    buffer: wgpu::Buffer,
    // Nb: View references &slice.
    view: &'a mut [MaybeUninit<u8>],
    // Needed to keep the binding alive.
    _binding: SliceBinding,
}

impl BytesBacking for WgpuAllocController<'_> {
    fn dealloc(&mut self) {
        let _ref = self.buffer.slice(..).get_mapped_range_mut();

        // We unmap the buffer and release the binding so that the same buffer can be used again.
        // Nb: This also resets the map context, so we don't need to drop the BufferViewMut at all.
        self.buffer.unmap();

        // The buffer will now return to the memory pool as the binding is dropped.
    }

    fn alloc_align(&self) -> usize {
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut *self.view
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        &*self.view
    }
}

impl<'a> WgpuAllocController<'a> {
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
    pub fn init(binding: SliceBinding, buffer: wgpu::Buffer) -> Self {
        let buf = buffer.clone();
        // acquire the view. We intentionally do _not_ drop the BufferViewMut.
        // Instead, when we unmap the buffer, it's all cleaned up properly.
        let range = ManuallyDrop::new(buffer.slice(..).get_mapped_range_mut());

        // SAFETY: This is only safe because we know exactly what BufferViewMut does.
        // range.as_ptr() reads the inner mapped ptr, which is valid for the lifetime of the buffer.
        // As we keep the buffer alive, this pointer is valid.
        //
        // SAFETY: We are converting a &mut [u8] to &mut [MaybeUninit<u8>].
        // This is safe because any u8 is a valid MaybeUninit<u8>.
        let u8_slice = unsafe {
            std::slice::from_raw_parts_mut(range.as_ptr() as *mut MaybeUninit<u8>, range.len())
        };

        Self {
            buffer: buf,
            view: u8_slice,
            _binding: binding,
        }
    }
}
