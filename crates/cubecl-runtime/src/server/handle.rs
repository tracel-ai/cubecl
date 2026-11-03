use alloc::sync::Arc;
use cubecl_common::device::ServiceId;
use cubecl_environment::stream::StreamId;
use cubecl_zspace::{Shape, Strides};

use crate::{
    memory_management::{
        ManagedMemoryBinding, ManagedMemoryHandle, ManagedMemoryId, WeakMemoryBinding,
    },
    server::{CopyDescriptor, TensorMapBinding},
};

/// How two handles' bytes relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlap {
    /// No byte in common.
    Disjoint,
    /// The same bytes.
    Same,
    /// Some bytes in common, but not all.
    Partial,
}

/// Server handle containing the [memory handle](crate::server::Handle).
pub struct Handle {
    /// Memory handle.
    pub memory: ManagedMemoryHandle,
    /// The service whose memory this handle addresses. A client checks it
    /// before handing the handle to its device: memory coordinates mean
    /// nothing on another device, and reading them there is not an error
    /// but garbage.
    pub service: ServiceId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
    /// The address kernels on other devices bind this memory at, once the
    /// owning client [shared](crate::client::Client::share_with_peers) it.
    pub peer: Option<PeerMemory>,
}

/// Memory one device lets kernels on the other devices of its runtime bind.
///
/// It holds the address the owning device resolved for the allocation, and a
/// binding that keeps the allocation in place there: a bound allocation is
/// never freed, handed out again or relocated, so the address stays valid for
/// as long as any clone lives.
///
/// Nothing orders the accesses. A kernel on another device reads whatever the
/// memory holds when it runs, and its writes land whenever they land. Callers
/// order them by synchronizing the devices, or by handing off inside the
/// kernels through system-scope atomics.
#[derive(Clone)]
pub struct PeerMemory {
    inner: Arc<PeerMemoryInner>,
}

struct PeerMemoryInner {
    /// Keeps the allocation in place on its device.
    _pin: ManagedMemoryBinding,
    /// The device address of the allocation's first byte.
    address: u64,
}

impl PeerMemory {
    /// Memory at `address` on its device, held in place by `pin`.
    pub fn new(pin: ManagedMemoryBinding, address: u64) -> Self {
        Self {
            inner: Arc::new(PeerMemoryInner { _pin: pin, address }),
        }
    }

    /// The device address of the allocation's first byte.
    pub fn address(&self) -> u64 {
        self.inner.address
    }
}

impl core::fmt::Debug for PeerMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PeerMemory")
            .field("address", &format_args!("{:#x}", self.inner.address))
            .finish()
    }
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.memory)
            .field("service", &self.service)
            .field("offset_start", &self.offset_start)
            .field("offset_end", &self.offset_end)
            .field("stream", &self.stream)
            .field("size", &self.size)
            .field("peer", &self.peer)
            .finish()
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            peer: self.peer.clone(),
        }
    }
}

impl Handle {
    /// Creates a new handle of the given size.
    pub fn from_memory(
        id: ManagedMemoryHandle,
        service: ServiceId,
        stream: StreamId,
        size: u64,
    ) -> Self {
        Self {
            memory: id,
            service,
            offset_start: None,
            offset_end: None,
            stream,
            size,
            peer: None,
        }
    }
    /// Creates a new handle of the given size.
    pub fn new(service: ServiceId, stream: StreamId, size: u64) -> Self {
        Self {
            memory: ManagedMemoryHandle::new(),
            service,
            offset_start: None,
            offset_end: None,
            stream,
            size,
            peer: None,
        }
    }
    /// Checks whether the handle can be mutated in-place without affecting other computation.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }

    /// Returns the [`BufferBinding`] corresponding to the current handle.
    pub fn binding(self) -> BufferBinding {
        BufferBinding {
            memory: self.memory.binding(),
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            peer: self.peer,
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
    /// How this handle's bytes relate to `other`'s.
    pub fn overlap(&self, other: &Self) -> Overlap {
        let bytes = |handle: &Self| {
            handle.offset_start.unwrap_or(0)..handle.size - handle.offset_end.unwrap_or(0)
        };
        let mine = bytes(self);
        let theirs = bytes(other);
        if self.memory.descriptor().id != other.memory.descriptor().id
            || mine.end <= theirs.start
            || theirs.end <= mine.start
        {
            Overlap::Disjoint
        } else if mine == theirs {
            Overlap::Same
        } else {
            Overlap::Partial
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

/// A resource passed to a kernel function
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum KernelResource {
    /// Buffer resource
    Buffer(BufferBinding),
    /// Tensor map resource for CUDA
    TensorMap(TensorMapBinding),
    /// Memory of another device of the same runtime, bound at its address
    /// there. The client turns a shared buffer of another device into this,
    /// so a server never sees a foreign [`Buffer`](Self::Buffer).
    Peer(PeerBinding),
}

impl KernelResource {
    /// The buffer of this device the resource binds, or `None` for memory
    /// of another device.
    pub fn local_binding(&self) -> Option<&BufferBinding> {
        match self {
            KernelResource::Buffer(binding) => Some(binding),
            KernelResource::TensorMap(map) => Some(&map.binding),
            KernelResource::Peer(_) => None,
        }
    }
}

/// A buffer of another device, as a kernel binds it.
#[derive(Clone, Debug)]
pub struct PeerBinding {
    /// The buffer on its own device. Holding it keeps the memory in place
    /// until the server drops it, once the kernel binding it completed.
    pub binding: BufferBinding,
    /// The device address of the bound bytes, offsets applied.
    pub address: u64,
}

impl PeerBinding {
    /// `binding` at the address its owner shared, or `None` when it was
    /// never shared.
    pub fn new(binding: BufferBinding) -> Option<Self> {
        let address = binding.peer.as_ref()?.address() + binding.offset_start.unwrap_or(0);
        Some(Self { binding, address })
    }

    /// The bytes bound.
    pub fn size(&self) -> u64 {
        self.binding.size_in_used()
    }
}

/// A buffer binding represents a [Handle] that is bound to managed memory.
///
/// The memory used is known by the compute server.
/// A buffer binding is only valid after being initlized with [`super::Server::initialize_bindings`]
///
/// # Notes
///
/// A buffer binding is detached from a [`Handle`], meaning that is won't affect [`Handle::can_mut`].
#[derive(Clone, Debug)]
pub struct BufferBinding {
    /// The id of the handle the binding is bound to.
    pub memory: ManagedMemoryBinding,
    /// The service whose memory this binding addresses; see [`Handle::service`].
    pub service: ServiceId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
    /// See [`Handle::peer`].
    pub peer: Option<PeerMemory>,
}

/// A [`BufferBinding`] that does not keep its memory reserved: what a record
/// of a buffer holds when holding the buffer would change what its pool may
/// reuse.
#[derive(Clone, Debug)]
pub struct WeakBufferBinding {
    memory: WeakMemoryBinding,
    service: ServiceId,
    offset_start: Option<u64>,
    offset_end: Option<u64>,
    stream: StreamId,
    size: u64,
}

impl WeakBufferBinding {
    /// The binding back, while its memory is allocated (see
    /// [`WeakMemoryBinding::upgrade`]).
    pub fn upgrade(&self) -> Option<BufferBinding> {
        Some(BufferBinding {
            memory: self.memory.upgrade()?,
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            peer: None,
        })
    }
}

impl BufferBinding {
    /// This binding, without keeping its memory reserved.
    pub fn downgrade(&self) -> WeakBufferBinding {
        WeakBufferBinding {
            memory: self.memory.downgrade(),
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }

    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }

    /// The byte range of the allocation this binding names: what the offsets
    /// leave of the buffer. This is the region the taint bookkeeping claims
    /// when work writing through this binding fails, and releases when work
    /// writing through it lands.
    pub fn range(&self) -> core::ops::Range<u64> {
        self.offset_start.unwrap_or(0)..self.size - self.offset_end.unwrap_or(0)
    }

    /// The identity a claim on this binding is stored under: the allocation
    /// and the byte range of it, exactly as [`range`](Self::range) computes
    /// it.
    ///
    /// Anything that deduplicates or compares claims — a capture's write set,
    /// above all — must key by this and nothing coarser. Two tensors carved
    /// from one batched allocation share a memory id and nothing else; keyed
    /// by the id alone, one sibling's claim swallows the others'.
    pub fn claim_key(&self) -> (ManagedMemoryId, u64, u64) {
        let range = self.range();
        (self.memory.id(), range.start, range.end)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}
