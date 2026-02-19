use cubecl_core::{
    server::{Allocation, AllocationDescriptor, Handle, HandleId, ServerAllocator},
    stream_id::StreamId,
    zspace::{Shape, Strides, strides},
};

pub struct WgpuAllocator {
    pub mem_aligment: usize,
}

impl ServerAllocator for WgpuAllocator {
    fn alloc(&self, stream_id: StreamId, descriptor: &AllocationDescriptor) -> Allocation {
        let strides = contiguous_strides(&descriptor.shape);
        let size = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let size = size.next_multiple_of(self.mem_aligment);
        let handle = Handle::new(HandleId::new(), None, None, stream_id, size as u64);

        Allocation::new(handle, strides)
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
