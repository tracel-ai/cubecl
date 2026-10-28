//! Allocation controller that lazily reads a device resource into host memory.

use super::Client;
use crate::server::Handle;
use alloc::boxed::Box;
use alloc::format;
use core::mem::MaybeUninit;
use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes,
};
use cubecl_environment::sync::{Mutex, Once};
use cubecl_zspace::{Shape, Strides, striding::has_contiguous_row_major_strides};

/// Allocation controller that lazily copies a device resource into host memory on first access.
///
/// Constructing the [`Bytes`] is cheap: it only captures the [`Client`], the source's [`Handle`]
/// and its layout. The device-to-host copy goes through the regular read path, including pinned
/// staging, and only happens the first time the bytes are read (e.g. during serialization); the
/// result is cached for subsequent accesses.
///
/// This lets a large number of device tensors be serialized without materializing them all in
/// host memory at once: each [`Bytes`] is read just-in-time and dropped right after.
///
/// # Semantics
///
/// Until the first access the controller holds the source's [`Handle`], so
/// [`Handle::can_mut`] is false for it and a caller that honours it does not write the buffer in
/// place: the data read is the data as it was at [`read_lazy`]. The handle is let go once the
/// bytes are on the host.
///
/// [`read_lazy`]: Client::read_lazy
pub struct LazyDeviceController {
    client: Client,
    /// The source until its bytes are on the host. A binding would keep the allocation alive but
    /// does not count toward [`Handle::can_mut`].
    source: Mutex<Option<Handle>>,
    shape: Shape,
    strides: Strides,
    elem_size: usize,
    /// Host bytes, materialized from the device on first access.
    materialized: Once<Bytes>,
}

impl LazyDeviceController {
    pub(super) fn new(
        client: Client,
        source: Handle,
        shape: Shape,
        strides: Strides,
        elem_size: usize,
    ) -> Self {
        Self {
            client,
            source: Mutex::new(Some(source)),
            shape,
            strides,
            elem_size,
            materialized: Once::new(),
        }
    }

    /// The [`Bytes`] this controller reads into on first access.
    pub(super) fn into_bytes(self) -> Bytes {
        let len = self.byte_len();
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
                    .expect("The source is held until its bytes are on the host");
                let descriptor = source.copy_descriptor(
                    self.shape.clone(),
                    self.strides.clone(),
                    self.elem_size,
                );
                cubecl_environment::future::reader::read_sync(
                    self.client.read_one_tensor_async(descriptor),
                )
                .map_err(|err| AccessError::Read(format!("{err:?}")))
            })?;
        self.source.lock().take();
        Ok(bytes)
    }

    /// The host byte length, derived from the layout without materializing.
    fn byte_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.elem_size
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

    // The host byte length, known from the layout without materializing.
    fn capacity(&self) -> usize {
        self.byte_len()
    }

    /// A zero-copy (until materialized) device sub-view over the byte range `[start, end)`:
    /// a new lazy controller that, when read, copies only those bytes off the device.
    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        // The source is let go once the bytes are on the host, which have no sub-view to take.
        let mut source = self.source.lock().clone()?;
        if start > end || end > self.byte_len() {
            return None;
        }
        // A byte sub-range only maps to a contiguous device region when the source is
        // contiguous (a strided/padded tensor has gaps the byte offsets can't express).
        if !has_contiguous_row_major_strides(&self.shape, &self.strides) {
            return None;
        }

        // Narrow the handle to the byte window `[start, end)`. Offsets are byte-level and
        // `offset_end` is trimmed-from-the-end (see `Binding::size_in_used`).
        let base = source.offset_start.unwrap_or(0);
        source.offset_end = Some(source.size - (base + end as u64));
        source.offset_start = Some(base + start as u64);

        Some(Box::new(LazyDeviceController::new(
            self.client.clone(),
            source,
            [end - start].into(),
            [1].into(),
            1,
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
        let slice: &mut [u8] = bytes;
        // SAFETY: same layout as above; every byte is initialized.
        Ok(unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) })
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        let bytes = self
            .ensure_init(AccessPolicy::default())
            .expect("device: host access failed");
        let len = buf.len().min(bytes.len());
        buf[..len].copy_from_slice(&bytes[..len]);
    }
}
