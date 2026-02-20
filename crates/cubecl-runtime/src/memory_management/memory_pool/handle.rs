use crate::memory_management::MemoryHandle;
use crate::server::MemorySlot;
use crate::{id::HandleRef, server::Handle};
use alloc::vec::Vec;
use cubecl_common::stream_id::StreamId;

#[doc = r" Memory Handle."]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedMemoryHandle {
    value: crate::id::HandleRef<ManagedMemoryId>,
}

#[doc = r" Memory ID."]
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ManagedMemoryId {
    pub(crate) value: usize,
}

impl ManagedMemoryHandle {
    #[doc = r" Create a new ID."]
    pub(crate) fn new() -> Self {
        let value = Self::gen_id();
        Self {
            value: crate::id::HandleRef::new(ManagedMemoryId { value }),
        }
    }
    fn gen_id() -> usize {
        static COUNTER: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if value == usize::MAX {
            core::panic!("Memory ID overflowed");
        }
        value
    }
}
impl core::ops::Deref for ManagedMemoryHandle {
    type Target = crate::id::HandleRef<ManagedMemoryId>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[doc = r" Binding of a memory handle."]
#[derive(Clone, Debug)]
pub struct ManagedMemoryBinding {
    value: crate::id::BindingRef<ManagedMemoryId>,
}
impl ManagedMemoryHandle {
    pub(crate) fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            value: self.value.binding(),
        }
    }
}
impl core::ops::Deref for ManagedMemoryBinding {
    type Target = crate::id::BindingRef<ManagedMemoryId>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl MemoryHandle<ManagedMemoryBinding> for ManagedMemoryHandle {
    fn can_mut(&self) -> bool {
        HandleRef::can_mut(self)
    }

    fn binding(self) -> ManagedMemoryBinding {
        self.binding()
    }
}

/// Take a list of sub-slices of a buffer and create a list of offset handles.
/// Sizes must be in bytes and handles will be aligned to the memory alignment.
pub fn partition_memory(
    memory: ManagedMemoryHandle,
    memory_size: u64,
    handles: &[Handle],
    cursor: u64,
    stream: StreamId,
) -> Vec<MemorySlot> {
    let mut offset = 0;
    let mut out = Vec::with_capacity(handles.len());

    for handle in handles {
        let size = handle.size();
        let buffer = MemorySlot {
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
