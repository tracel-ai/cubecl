use crate::compute::device_poison::PoisonWatch;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes,
};
use cubecl_environment::sync::Mutex;
use cubecl_server::memory_management::ManagedMemoryBinding;
use wgpu::BufferView;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    view: Option<BufferView>,
    buffer: wgpu::Buffer,
    _binding: ManagedMemoryBinding,
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

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    fn may_alias_mutable_storage(&self) -> bool {
        false
    }

    // A mapped wgpu staging buffer is always host-resident: the policy never forces a copy here.
    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        // - Caller promises not to write uninitialized values.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(bytes.as_ptr() as *mut MaybeUninit<u8>, bytes.len())
        })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[std::mem::MaybeUninit<u8>], AccessError> {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        Ok(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const MaybeUninit<u8>, bytes.len())
        })
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
    pub fn init(binding: ManagedMemoryBinding, buffer: wgpu::Buffer) -> Self {
        // Needs immutable as of wgpu v29, mutable doesn't allow dereferencing as slice.
        // This only affects wgpu's internal overlap checks so it should be fine as long as we
        // map the whole buffer anyways.
        let buf_view = buffer.get_mapped_range(..).unwrap();

        Self {
            view: Some(buf_view),
            buffer,
            _binding: binding,
        }
    }
}

/// A staging buffer whose map was requested. Dropped before
/// [`into_bytes`](Self::into_bytes), it sees the buffer unmapped before its
/// pool can hand it out again.
pub(crate) struct MappingStaging {
    staging: Arc<Mutex<Staging>>,
    size: usize,
    poison: PoisonWatch,
}

/// What the read and its map's callback share. Only the callback may unmap a
/// buffer whose map is still completing: wgpu reports it unmapped meanwhile.
struct Staging {
    buffer_and_binding: Option<(wgpu::Buffer, ManagedMemoryBinding)>,
    mapped: bool,
    abandoned: bool,
}

impl MappingStaging {
    /// Map `buffer` for reading, calling `on_mapped` once the map completes.
    pub(crate) fn map(
        buffer: wgpu::Buffer,
        binding: ManagedMemoryBinding,
        size: usize,
        poison: &PoisonWatch,
        on_mapped: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) -> Self {
        let staging = Arc::new(Mutex::new(Staging {
            buffer_and_binding: Some((buffer.clone(), binding)),
            mapped: false,
            abandoned: false,
        }));
        let shared = staging.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                shared.lock().complete(result.is_ok());
                on_mapped(result);
            });

        Self {
            staging,
            size,
            poison: poison.clone(),
        }
    }

    /// The buffer's first `size` bytes, once its map has completed.
    pub(crate) fn into_bytes(self) -> Bytes {
        let (buffer, binding) = self
            .staging
            .lock()
            .buffer_and_binding
            .take()
            .expect("taken once");
        let controller = Box::new(WgpuAllocController::init(binding, buffer));
        // SAFETY: The binding has initialized memory for at least `size` bytes.
        unsafe { Bytes::from_controller(controller, self.size) }
    }
}

impl Drop for MappingStaging {
    fn drop(&mut self) {
        let mut staging = self.staging.lock();
        staging.abandon();
        // A lost device may never run the callback, which would keep the buffer reserved.
        let lost = match self.poison.is_poisoned() {
            true => staging.buffer_and_binding.take(),
            false => None,
        };
        // Dropped unlocked: dropping a buffer's last handle runs its pending callback.
        core::mem::drop(staging);
        core::mem::drop(lost);
    }
}

impl Staging {
    fn complete(&mut self, mapped: bool) {
        self.mapped = mapped;
        if self.abandoned {
            self.release();
        }
    }

    fn abandon(&mut self) {
        self.abandoned = true;
        if self.mapped {
            self.release();
        }
    }

    /// Unmap the buffer if it is mapped, then hand it back to its pool.
    fn release(&mut self) {
        if let Some((buffer, _binding)) = self.buffer_and_binding.take()
            && self.mapped
        {
            buffer.unmap();
        }
    }
}
