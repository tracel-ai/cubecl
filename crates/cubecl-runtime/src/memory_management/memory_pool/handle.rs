use crate::memory_id_type;
use crate::memory_management::MemoryHandle;
use crate::{id::HandleRef, server::Handle};
use alloc::vec::Vec;

// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

impl MemoryHandle<SliceBinding> for SliceHandle {
    fn can_mut(&self) -> bool {
        HandleRef::can_mut(self)
    }

    fn binding(self) -> SliceBinding {
        self.binding()
    }
}

/// Take a list of sub-slices of a buffer and create a list of offset handles.
/// Sizes must be in bytes and aligned to the memory alignment.
pub fn offset_handles(base_handle: Handle, sizes_bytes: &[usize]) -> Vec<Handle> {
    let total_size: usize = sizes_bytes.iter().sum();
    let mut offset = 0;
    let mut out = Vec::new();

    for size in sizes_bytes {
        let handle = base_handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((total_size - offset - size) as u64);
        out.push(handle);
        offset += size;
    }

    out
}
