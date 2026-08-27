use crate::{
    memory_management::optimal_align,
    server::{
        Handle, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy, MemoryLayoutStrategy,
    },
};
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;
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
        let mut total_size = 0u64;

        let (sizes, strides): (Vec<_>, Vec<_>) = descriptors
            .iter()
            .map(|descriptor| {
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
        let handle = base_handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((total_size - offset - size) as u64);
        out.push(handle);
        offset += size.next_multiple_of(buffer_align);
    }

    out
}

/// The 2D geometry of a copy to or from a pitched allocation: how wide each
/// row is, how many rows there are, and the stride between their starts.
///
/// [`of`](Self::of) answers `None` for an allocation that needs no 2D copy at
/// all. A row stride equal to the row width means no padding, so the whole
/// buffer is one contiguous span and the plain linear copy is both correct and
/// faster — drivers also refuse the 2D form for very tall transfers, an
/// embedding table's 128k rows say, which the linear path handles.
///
/// This is the read side of what [`PitchedMemoryLayoutPolicy`] produces, and
/// it is the same arithmetic whichever driver performs the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pitch {
    /// The bytes in one row, which is what both sides of the copy transfer.
    pub width_bytes: usize,
    /// How many rows the copy moves.
    pub height: usize,
    /// The bytes between the starts of two rows on the pitched side.
    pub stride_bytes: usize,
}

impl Pitch {
    /// The pitch of an allocation with this layout, or `None` when its rows
    /// are contiguous.
    ///
    /// The caller has already validated the strides as pitched row-major; this
    /// only asks whether there is padding to step over.
    pub fn of(shape: &[usize], strides: &[usize], elem_size: usize) -> Option<Self> {
        let rank = shape.len();
        let width = *shape.last().unwrap_or(&1);
        if rank < 2 || strides[rank - 2] == width {
            return None;
        }
        Some(Self {
            width_bytes: width * elem_size,
            height: shape.iter().rev().skip(1).product(),
            stride_bytes: strides[rank - 2] * elem_size,
        })
    }
}
