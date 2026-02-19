use crate::memory_id_type;
use crate::memory_management::MemoryHandle;
use crate::server::Buffer;
use crate::{id::HandleRef, server::Handle};
use alloc::vec::Vec;
use cubecl_common::stream_id::StreamId;

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
pub fn create_buffers(
    memory: SliceHandle,
    memory_size: u64,
    handles: &[Handle],
    cursor: u64,
    stream: StreamId,
) -> Vec<Buffer> {
    let mut offset = 0;
    let mut out = Vec::with_capacity(handles.len());

    for handle in handles {
        let size = handle.size();
        let buffer = Buffer {
            memory: memory.clone(),
            offset_start: Some(offset),
            offset_end: Some(memory_size - offset - size),
            cursor,
            stream,
            size,
        };
        out.push(buffer);
        offset += size;
    }

    out
}

/// Calculates a best-effort heuristic for the alignment of row-aligned tensors.
/// Prefers contiguous alignments for unit dimensions, 16-byte minimum alignment for non-unit,
/// scaling with input size up to `buffer_align`.
pub fn optimal_align(shape: usize, elem_size: usize, buffer_align: usize) -> usize {
    if shape == 1 {
        elem_size
    } else {
        (shape * elem_size)
            .next_power_of_two()
            .clamp(16, buffer_align)
    }
}
