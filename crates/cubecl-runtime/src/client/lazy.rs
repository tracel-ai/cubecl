//! Allocation controller that lazily reads a device resource into host memory.

use super::{Client, ClosedAsError};
use crate::server::{CopyDescriptor, Handle, LazyRead, LazyReadHold, ServerError};
use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use core::ops::Range;
use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, Writer,
};
use cubecl_environment::sync::{Mutex, Once};
use cubecl_zspace::striding::has_contiguous_row_major_strides;

/// Allocation controller that lazily copies a device resource into host memory on first access.
///
/// Constructing the [`Bytes`] is cheap: it only captures the [`Client`] and the source's
/// [`CopyDescriptor`]. The device-to-host copy goes through the regular read path, including
/// pinned staging, and only happens the first time the bytes are read (e.g. during
/// serialization); the result is cached for subsequent accesses.
///
/// This lets a large number of device tensors be serialized without materializing them all in
/// host memory at once: each [`Bytes`] is read just-in-time and dropped right after.
///
/// Until the first access the controller holds the source's [`Handle`], so
/// [`Handle::can_mut`] is false for it and a caller that honours it does not write the buffer in
/// place, which spares the server a copy. The source is let go once the host bytes are
/// independent of it. A zero-copy host read retains the handle until dropped or copied for
/// mutation.
///
/// # Assumptions
///
/// - Nothing writes the buffer through a pointer [`Client::get_resource`] handed out: the server
///   never sees such a write.
///
/// # Guarantees
///
/// - On a server whose reads copy, the bytes are the buffer as it was at [`read_lazy`], under
///   every write that server queues before the first access: a kernel, a host write, a graph's
///   replay or a collective starts the read first. On one whose reads alias its memory, the CPU,
///   they are as live as an eager read's.
///
/// [`read_lazy`]: Client::read_lazy
pub struct LazyDeviceController {
    client: Client,
    /// The source until the host bytes no longer alias it.
    source: Mutex<Option<CopyDescriptor<Handle>>>,
    /// The bytes of the read the server keeps that these are, shared with every view of the
    /// same bytes. Let go once the host bytes are read.
    read: Mutex<Option<LazyReadHold>>,
    /// The host byte length, known without the source.
    len: usize,
    /// Host bytes, materialized from the device on first access.
    materialized: Once<Bytes>,
}

impl LazyDeviceController {
    pub(super) fn new(client: Client, source: CopyDescriptor<Handle>, read: LazyReadHold) -> Self {
        Self {
            client,
            len: read.range().len(),
            source: Mutex::new(Some(source)),
            read: Mutex::new(Some(read)),
            materialized: Once::new(),
        }
    }

    /// The [`Bytes`] this controller reads into on first access.
    pub(super) fn into_bytes(self) -> Bytes {
        let len = self.len;
        // SAFETY: the controller materializes exactly `len` bytes on first access.
        unsafe { Bytes::from_controller(Box::new(self), len) }
    }

    /// Materialize the device resource into host memory on first call, returning the cached
    /// [`Bytes`] afterwards. Thread-safe: concurrent first accesses read exactly once.
    ///
    /// Honors `policy`: if not yet materialized and the policy forbids copies, returns
    /// [`AccessError::WouldCopy`]. A failed device read returns [`AccessError::Read`] and leaves
    /// the cell uninitialized (retryable).
    fn ensure_init(&self, policy: AccessPolicy) -> Result<&Bytes, AccessError> {
        if let Some(bytes) = self.materialized.get() {
            return Ok(bytes);
        }
        if !policy.copy_allowed() {
            return Err(AccessError::WouldCopy);
        }

        let bytes = self
            .materialized
            .try_call_once(|| -> Result<Bytes, AccessError> {
                let source = self
                    .source
                    .lock()
                    .clone()
                    .expect("The source should be held until its bytes are on the host");
                let (read, range) = self
                    .read
                    .lock()
                    .as_ref()
                    .map(|held| (held.read().clone(), held.range()))
                    .expect("The read should be held until its bytes are on the host");
                self.client
                    .read_lazily(source.binding(), &read, range)
                    .map_err(|err| AccessError::Read(format!("{err:?}")))
            })?;
        self.read.lock().take();
        if !bytes.may_alias_mutable_storage() {
            self.source.lock().take();
        }
        Ok(bytes)
    }
}

impl AllocationController for LazyDeviceController {
    fn alloc_align(&self) -> usize {
        // Avoid materializing just to answer. `try_detach` is never implemented, so this is not
        // used for `Vec` reconstruction; a conservative value is fine before materialization.
        match self.materialized.get() {
            Some(bytes) => bytes.align(),
            None => core::mem::align_of::<u128>(),
        }
    }

    fn property(&self) -> AllocationProperty {
        match self.materialized.get() {
            Some(bytes) => bytes.property(),
            None => AllocationProperty::Device,
        }
    }

    fn may_alias_mutable_storage(&self) -> bool {
        match self.materialized.get() {
            Some(bytes) => bytes.may_alias_mutable_storage(),
            None => true,
        }
    }

    // The host byte length, known without materializing.
    fn capacity(&self) -> usize {
        self.len
    }

    /// A zero-copy (until materialized) device sub-view over the byte range `[start, end)`:
    /// a new lazy controller that, when read, copies only those bytes off the device.
    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        // Aliasing host bytes can still hold the source; views must not bypass cached data.
        if self.materialized.get().is_some() {
            return None;
        }
        let source = self.source.lock().clone()?;
        if start > end || end > self.len {
            return None;
        }
        // A byte sub-range only maps to a contiguous device region when the source is
        // contiguous (a strided/padded tensor has gaps the byte offsets can't express).
        if !has_contiguous_row_major_strides(&source.shape, &source.strides) {
            return None;
        }

        // Narrow the handle to the byte window `[start, end)`. Offsets are byte-level and
        // `offset_end` is trimmed-from-the-end (see `Binding::size_in_used`).
        let mut handle = source.handle;
        let base = handle.offset_start.unwrap_or(0);
        handle.offset_end = Some(handle.size - (base + end as u64));
        handle.offset_start = Some(base + start as u64);

        // Flat byte descriptor over the sub-range; materializing reads only these bytes.
        let descriptor = CopyDescriptor::new(handle, [end - start].into(), [1].into(), 1);
        let read = self.read.lock().as_ref()?.narrowed(start..end);
        Some(Box::new(LazyDeviceController::new(
            self.client.clone(),
            descriptor,
            read,
        )))
    }

    fn memory(&self, policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        let bytes = self.ensure_init(policy)?;
        let slice: &[u8] = bytes;
        // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` share a layout, and every byte read from the
        // device is initialized.
        Ok(unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) })
    }

    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        self.ensure_init(policy)?;

        // SAFETY: `ensure_init` guarantees the cell is initialized, and `&mut self` is exclusive.
        let bytes = self
            .materialized
            .get_mut()
            .expect("materialized must be set after init");
        let writer = match policy.copy_allowed() {
            true => Writer::new(),
            false => Writer::new().no_copy(),
        };
        let slice = bytes.write(writer)?;
        let (ptr, len) = (slice.as_mut_ptr(), slice.len());
        if !bytes.may_alias_mutable_storage() {
            self.source.lock().take();
        }
        // SAFETY: every byte is initialized. Querying aliasing does not invalidate the
        // pointer, and the source is released only when the host bytes are independent.
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.cast(), len) })
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        let bytes = self
            .ensure_init(AccessPolicy::default())
            .expect("device: host access failed");
        let len = buf.len().min(bytes.len());
        buf[..len].copy_from_slice(&bytes[..len]);
    }
}

impl Client {
    /// The bytes `source` names, taken from `range` of what a write started `read` for when one
    /// did, or else read now, with work writing them afterwards waiting for that copy.
    ///
    /// Asked on the device thread, so a write queued before this access has started the read.
    ///
    /// # Errors
    ///
    /// The server's refusal to keep `read`, and the read's own errors.
    fn read_lazily(
        &self,
        source: CopyDescriptor,
        read: &Arc<LazyRead>,
        range: Range<usize>,
    ) -> Result<Bytes, ServerError> {
        let stream_id = self.stream_id();
        let started = read.clone();
        let now = self
            .device
            .submit_blocking(move |server| -> Result<_, ServerError> {
                Ok(match started.is_started()? {
                    true => None,
                    false => Some(server.read_lazy_bytes(started, source, stream_id)),
                })
            })
            .closed_as_error()?;
        match now {
            Some(now) => Ok(cubecl_environment::future::reader::read_sync(now)?.remove(0)),
            None => read.started_bytes(range),
        }
    }
}
