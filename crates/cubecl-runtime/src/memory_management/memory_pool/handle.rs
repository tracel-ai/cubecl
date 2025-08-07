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
/// Sizes must be in bytes and handles will be aligned to the memory alignment.
pub fn offset_handles(
    base_handle: Handle,
    sizes_bytes: &[usize],
    buffer_align: usize,
) -> Vec<Handle> {
    let total_size = base_handle.size() as usize;
    let mut offset = 0;
    let mut out = Vec::new();

    for size in sizes_bytes {
        let handle = base_handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((total_size - offset - size) as u64);
        out.push(handle);
        offset += size.next_multiple_of(buffer_align);
    }

    out
}
