use core::mem::MaybeUninit;
use core::pin::Pin;
use cubecl_common::bytes::BytesBacking;
use cubecl_runtime::memory_management::SliceBinding;
use wgpu::BufferViewMut;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController<'a> {
    // IMPORTANT: Field order matters for drop order!
    // The view must be dropped before the buffer since it borrows from it.
    view: Option<BufferViewMut<'a>>,
    buffer: Pin<Box<wgpu::Buffer>>,
    _binding: SliceBinding,
}

impl BytesBacking for WgpuAllocController<'_> {
    fn dealloc(&mut self) {
        // Drop the view first, then unmap the buffer.
        // This ensures proper cleanup order since the view borrows from the buffer.
        drop(self.view.take());

        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.buffer.unmap();

        // The buffer will now return to the memory pool as the binding is dropped.
    }

    fn alloc_align(&self) -> usize {
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let bytes: &mut [u8] = self.view.as_mut().unwrap();
        // SAFETY: MaybeUninit<u8> has the same layout as u8
        unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut MaybeUninit<u8>, bytes.len())
        }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY: MaybeUninit<u8> has the same layout as u8
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
        let buf = Pin::new(Box::new(buffer));

        // SAFETY: We're extending the lifetime to match the controller's lifetime, which is safe because:
        // 1. The view is always dropped before the buffer (via ManuallyDrop in dealloc)
        // 2. The buffer stays alive as long as the controller exists
        // 3. The buffer is pinned and will never move after creating the view
        // 4. BufferSlice holds &wgpu::Buffer, so the buffer address must remain stable
        let view = unsafe {
            std::mem::transmute::<BufferViewMut<'_>, BufferViewMut<'a>>(
                buf.slice(..).get_mapped_range_mut(),
            )
        };

        Self {
            view: Some(view),
            buffer: buf,
            _binding: binding,
        }
    }
}
