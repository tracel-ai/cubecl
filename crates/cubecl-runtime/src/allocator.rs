use cubecl_common::stream_id::StreamId;
use cubecl_zspace::{Shape, Strides, strides};

use crate::{
    client::ComputeClient,
    memory_management::optimal_align,
    runtime::Runtime,
    server::{
        Handle, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy, MemoryLayoutStrategy,
    },
};

/// Allocators where every allocations is with contiguous memory.
pub struct ContiguousMemoryLayoutPolicy {
    mem_alignment: usize,
}

/// Allocators where some allocations can leverage a pitched layout.
pub struct PitchedMemoryLayoutPolicy {
    mem_alignment: usize,
}

impl MemoryLayoutPolicy for PitchedMemoryLayoutPolicy {
    fn apply<R: Runtime>(
        &self,
        client: ComputeClient<R>,
        stream_id: StreamId,
        descriptor: &MemoryLayoutDescriptor,
    ) -> MemoryLayout<R> {
        let last_dim = descriptor.shape.last().copied().unwrap_or(1);
        let pitch_align = match descriptor.strategy {
            MemoryLayoutStrategy::Contiguous => 1,
            MemoryLayoutStrategy::Optimized => {
                optimal_align(last_dim, descriptor.elem_size, self.mem_alignment)
            }
        };

        let rank = descriptor.shape.len();
        let width = *descriptor.shape.last().unwrap_or(&1);
        let height: usize = descriptor.shape.iter().rev().skip(1).product();
        let height = Ord::max(height, 1);
        let width_bytes = width * descriptor.elem_size;
        let pitch = width_bytes.next_multiple_of(pitch_align);
        let size = height * pitch;
        let mut strides = strides![1; rank];

        if rank > 1 {
            strides[rank - 2] = pitch / descriptor.elem_size;
        }
        if rank > 2 {
            for i in (0..rank - 2).rev() {
                strides[i] = strides[i + 1] * descriptor.shape[i + 1];
            }
        }

        let handle = Handle::new(client, stream_id, size as u64);

        MemoryLayout::new(handle, strides)
    }
}

impl ContiguousMemoryLayoutPolicy {
    /// Creates a new allocator with the given memory aligment.
    pub fn new(mem_alignment: usize) -> Self {
        Self { mem_alignment }
    }
}

impl PitchedMemoryLayoutPolicy {
    /// Creates a new allocator with the given memory aligment.
    pub fn new(mem_aligment: usize) -> Self {
        Self {
            mem_alignment: mem_aligment,
        }
    }
}

impl MemoryLayoutPolicy for ContiguousMemoryLayoutPolicy {
    fn apply<R: Runtime>(
        &self,
        client: ComputeClient<R>,
        stream_id: StreamId,
        descriptor: &MemoryLayoutDescriptor,
    ) -> MemoryLayout<R> {
        let strides = contiguous_strides(&descriptor.shape);
        let size_before = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let size_after = size_before.next_multiple_of(self.mem_alignment);

        let handle = Handle::new(client, stream_id, size_after as u64);

        let handle = match size_after > size_before {
            true => handle.offset_end((size_after - size_before) as u64),
            false => handle,
        };

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
