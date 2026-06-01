use std::println;

use crate::{
    memory_management::optimal_align,
    server::{
        Handle, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy, MemoryLayoutStrategy,
    },
};
use alloc::vec::Vec;
use cubecl_common::stream_id::StreamId;
use cubecl_zspace::{Shape, Strides, strides};

/// Allocators where every allocations is with contiguous memory.
pub struct ContiguousMemoryLayoutPolicy {
    mem_alignment: usize,
}

/// Allocators where some allocations can leverage a pitched layout.
pub struct PitchedMemoryLayoutPolicy {
    mem_alignment: usize,
}

impl MemoryLayoutPolicy for PitchedMemoryLayoutPolicy {
    fn apply(
        &self,
        stream_id: StreamId,
        descriptors: &[MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>) {
        println!("=============================================");
        println!("In pitched");
        let mut total_size = 0u64;

        let (sizes, strides): (Vec<_>, Vec<_>) = descriptors
            .iter()
            .map(|descriptor| {
                println!("descriptor: {descriptor:?}");
                let last_dim = descriptor.shape.last().copied().unwrap_or(1);
                println!("last_dim: {last_dim}");
                let pitch_align = match descriptor.strategy {
                    MemoryLayoutStrategy::Contiguous => 1,
                    MemoryLayoutStrategy::Optimized => {
                        optimal_align(last_dim, descriptor.elem_size, self.mem_alignment)
                    }
                };
                println!("pitch_align: {pitch_align}");

                let rank = descriptor.shape.len();
                let width = *descriptor.shape.last().unwrap_or(&1);
                println!("rank: {rank}");
                println!("width: {width}");
                let height: usize = descriptor.shape.iter().rev().skip(1).product();
                println!("height1: {height}");
                let height = Ord::max(height, 1);
                println!("height2: {height}");

                let width_bytes = width * descriptor.elem_size;
                println!("width_bytes: {width_bytes}");
                let pitch = width_bytes.next_multiple_of(pitch_align);
                println!("height: {height}");
                println!("pitch: {pitch}");
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
                total_size += size.next_multiple_of(self.mem_alignment) as u64;
                (size, strides)
            })
            .unzip();

        let base_handle = Handle::new(stream_id, total_size);

        let layouts = offset_handles(base_handle.clone(), &sizes, self.mem_alignment)
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| MemoryLayout::new(handle, strides))
            .collect();
        println!("=============================================");
        (base_handle, layouts)
    }
}

impl ContiguousMemoryLayoutPolicy {
    /// Creates a new allocator with the given memory alignment.
    pub fn new(mem_alignment: usize) -> Self {
        Self { mem_alignment }
    }
}

impl PitchedMemoryLayoutPolicy {
    /// Creates a new allocator with the given memory alignment.
    pub fn new(mem_alignment: usize) -> Self {
        Self { mem_alignment }
    }
}

impl MemoryLayoutPolicy for ContiguousMemoryLayoutPolicy {
    fn apply(
        &self,
        stream_id: StreamId,
        descriptors: &[MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>) {
        println!("In contiguous");
        let mut total_size = 0u64;
        let (sizes, strides): (Vec<_>, Vec<_>) = descriptors
            .iter()
            .map(|desc| {
                let size = desc.shape.iter().product::<usize>() * desc.elem_size;
                total_size += size.next_multiple_of(self.mem_alignment) as u64;
                (size, contiguous_strides(&desc.shape))
            })
            .unzip();

        let base_handle = Handle::new(stream_id, total_size);

        let layouts = offset_handles(base_handle.clone(), &sizes, self.mem_alignment)
            .into_iter()
            .zip(strides)
            .map(|(handle, stride)| MemoryLayout::new(handle, stride))
            .collect();

        (base_handle, layouts)
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
        println!("offset_handles totaal_size: {total_size}");
        println!("offset_handles offset: {offset}");
        println!("offset_handles size: {size}");
        let handle = base_handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((total_size - offset - size) as u64);
        out.push(handle);
        offset += size.next_multiple_of(buffer_align);
    }

    out
}
