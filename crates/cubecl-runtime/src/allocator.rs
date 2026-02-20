use cubecl_common::stream_id::StreamId;
use cubecl_zspace::{Shape, Strides, strides};

use crate::server::{Handle, HandleId, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy};

/// Allocators where every allocations is with contiguous memory.
pub struct ContiguousMemoryLayoutPolicy {
    mem_aligment: usize,
}

impl ContiguousMemoryLayoutPolicy {
    /// Creates a new allocator with the given memory aligment.
    pub fn new(mem_aligment: usize) -> Self {
        Self { mem_aligment }
    }
}

impl MemoryLayoutPolicy for ContiguousMemoryLayoutPolicy {
    fn apply(&self, stream_id: StreamId, descriptor: &MemoryLayoutDescriptor) -> MemoryLayout {
        let strides = contiguous_strides(&descriptor.shape);
        let size_before = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let size_after = size_before.next_multiple_of(self.mem_aligment);
        let offset_end = match size_after > size_before {
            true => Some((size_after - size_before) as u64),
            false => None,
        };
        let handle = Handle::new(
            HandleId::new(),
            None,
            offset_end,
            stream_id,
            size_after as u64,
        );

        MemoryLayout::new(handle, strides)
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
