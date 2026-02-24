use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use alloc::sync::Arc;
use cubecl_common::stream_id::StreamId;

use crate::{client::ComputeClient, runtime::Runtime};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// An handle that points to memory.
pub struct HandleId {
    value: u64,
    count: Arc<()>,
}

static HANDLE_COUNT: AtomicU64 = AtomicU64::new(0);

impl HandleId {
    /// Creates a new id.
    pub fn new() -> Self {
        let value = HANDLE_COUNT.fetch_add(1, Ordering::Acquire);
        Self {
            value,
            count: Arc::new(()),
        }
    }
    /// Checks wheter the current handle can be mutated.
    pub fn can_mut(&self) -> bool {
        // One reference by the server/queue.
        Arc::strong_count(&self.count) <= 2
    }
    /// Checks wheter the current handle is free.
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.count) == 1
    }
}

/// Server handle containing the [memory handle](crate::server::Handle).
#[derive(new, Debug, PartialEq, Eq)]
pub struct Handle {
    /// Memory handle.
    pub id: HandleId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    // pub cursor: u64,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
}

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
/// An handle that points to memory.
pub struct HandleId2 {
    value: u64,
}

/// Server handle containing the [memory handle](crate::server::Handle).
pub struct Handle2<R: Runtime> {
    /// Memory handle.
    pub(crate) id: HandleId2,
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

impl<R: Runtime> Drop for Handle2<R> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Acquire);

        if count <= 1 {
            self.client.free(todo!());
        }
    }
}

impl<R: Runtime> Clone for Handle2<R> {
    fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Acquire);

        Self {
            count: self.count.clone(),
            id: self.id.clone(),
            offset_start: self.offset_start.clone(),
            offset_end: self.offset_end.clone(),
            stream: self.stream.clone(),
            size: self.size.clone(),
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> Handle2<R> {
    pub fn can_mut(self) -> bool {
        let count = self.count.load(Ordering::Acquire);
        count <= 1
    }

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
            free,
        }
    }
}

/// A binding is detached from a [Handle], meaning that is won't affect [Handle::can_mut].
pub struct Binding {
    pub(crate) id: HandleId2,
    /// Memory offset in bytes.
    pub(crate) offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub(crate) offset_end: Option<u64>,
    /// The stream where the data was created.
    pub(crate) stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
    /// Wheter the binding is used for the last time.
    pub(crate) free: bool,
}
