use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use alloc::sync::Arc;
use cubecl_common::stream_id::StreamId;
use cubecl_zspace::{Shape, Strides};

use crate::{client::ComputeClient, runtime::Runtime, server::CopyDescriptor};

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
/// An handle that points to memory.
pub struct HandleId {
    value: u64,
}

static HANDLE_COUNT: AtomicU64 = AtomicU64::new(0);

impl Default for HandleId {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleId {
    /// Creates a new id.
    pub fn new() -> Self {
        let value = HANDLE_COUNT.fetch_add(1, Ordering::Relaxed);
        Self { value }
    }
}

/// Server handle containing the [memory handle](crate::server::Handle).
pub struct Handle<R: Runtime> {
    /// Memory handle.
    pub(crate) id: HandleId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
    pub(crate) count: Arc<AtomicU32>,
    /// The compute client.
    pub client: ComputeClient<R>,
}

impl<R: Runtime> core::fmt::Debug for Handle<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.id)
            .field("offset_start", &self.offset_start)
            .field("offset_end", &self.offset_end)
            .field("stream", &self.stream)
            .field("size", &self.size)
            .finish()
    }
}

impl<R: Runtime> Drop for Handle<R> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Acquire);

        if count <= 1 {
            self.client.free(self.id, self.stream);
        }
    }
}

impl<R: Runtime> Clone for Handle<R> {
    fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Acquire);

        Self {
            count: self.count.clone(),
            id: self.id,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> Handle<R> {
    /// Creates a new handle of the given size.
    pub fn new(client: ComputeClient<R>, stream: StreamId, size: u64) -> Self {
        Self {
            id: HandleId::new(),
            offset_start: None,
            offset_end: None,
            stream,
            size,
            count: Arc::new(AtomicU32::new(1)),
            client,
        }
    }
    /// Checks wheter the handle can be mutated in-place without affecting other computation.
    pub fn can_mut(&self) -> bool {
        let count = self.count.load(Ordering::Acquire);
        count <= 1
    }

    /// Returns the [`Binding`] corresponding to the current handle.
    pub fn binding(self) -> Binding {
        let count = self.count.load(Ordering::Acquire);
        let free = count <= 1;

        if free {
            // Avoids an unwanted drop on the same thread.
            //
            // Since `drop` is called after `into_ir`, we must not register a drop if the tensor
            // was consumed with a `ReadWrite` status.
            self.count.fetch_add(1, Ordering::Acquire);
        }

        Binding {
            id: self.id,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            last_use: free,
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
    pub id: HandleId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
    /// Wheter the binding is used for the last time.
    pub last_use: bool,
}

impl Binding {
    /// Creates a new binding manually.
    ///
    /// # Warning
    ///
    /// Using this method means you have to manually cleanup the binding with [`super::ComputeServer::free`].
    /// This should only be used `inside` the server, if you want to create a new handle and aren't
    /// implementing a server, use [`ComputeClient::create`] instead.
    pub fn new_manual(stream: StreamId, size: u64, single_use: bool) -> Self {
        Self {
            id: HandleId::new(),
            offset_start: None,
            offset_end: None,
            stream,
            size,
            last_use: single_use,
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
