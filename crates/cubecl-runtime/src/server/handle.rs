use cubecl_common::stream_id::StreamId;
use cubecl_zspace::{Shape, Strides};

use crate::{
    memory_management::{ManagedMemoryBinding, ManagedMemoryHandle},
    server::CopyDescriptor,
};

/// Server handle containing the [memory handle](crate::server::Handle).
pub struct Handle {
    /// Memory handle.
    pub memory: ManagedMemoryHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.memory)
            .field("offset_start", &self.offset_start)
            .field("offset_end", &self.offset_end)
            .field("stream", &self.stream)
            .field("size", &self.size)
            .finish()
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }
}

impl Handle {
    /// Creates a new handle of the given size.
    pub fn from_memory(id: ManagedMemoryHandle, stream: StreamId, size: u64) -> Self {
        Self {
            memory: id,
            offset_start: None,
            offset_end: None,
            stream,
            size,
        }
    }
    /// Creates a new handle of the given size.
    pub fn new(stream: StreamId, size: u64) -> Self {
        Self {
            memory: ManagedMemoryHandle::new(),
            offset_start: None,
            offset_end: None,
            stream,
            size,
        }
    }
    /// Checks whether the handle can be mutated in-place without affecting other computation.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }

    /// Returns the [`Binding`] corresponding to the current handle.
    pub fn binding(self) -> Binding {
        Binding {
            memory: self.memory.binding(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }

    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }

    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn copy_descriptor(
        self,
        shape: Shape,
        strides: Strides,
        elem_size: usize,
    ) -> CopyDescriptor {
        CopyDescriptor {
            shape,
            strides,
            elem_size,
            handle: self.binding(),
        }
    }
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// A binding represents a [Handle] that is bound to managed memory.
///
/// The memory used is known by the compute server.
/// A binding is only valid after being initlized with [`super::ComputeServer::initialize_bindings`]
///
/// # Notes
///
/// A binding is detached from a [`Handle`], meaning that is won't affect [`Handle::can_mut`].
#[derive(Clone, Debug)]
pub struct Binding {
    /// The id of the handle the binding is bound to.
    pub memory: ManagedMemoryBinding,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
}

impl Binding {
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}
