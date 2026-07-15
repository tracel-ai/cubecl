//! Allocation controller that lazily reads a device resource into host memory.

use super::ComputeClient;
use crate::runtime::Runtime;
use crate::server::CopyDescriptor;
use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes,
};
use cubecl_zspace::striding::has_contiguous_row_major_strides;
use spin::Once;

/// Allocation controller that lazily copies a device resource into host memory on first access.
///
/// Constructing the [`Bytes`] is cheap: it only captures the [`ComputeClient`] and a
/// [`CopyDescriptor`], whose [`Binding`](crate::server::Binding) keeps the device allocation
/// alive. The device-to-host copy — going through the regular read path, including pinned
/// staging — only happens the first time the bytes are read (e.g. during serialization), and
/// the result is cached for subsequent accesses.
///
/// This lets a large number of device tensors be serialized without materializing them all in
/// host memory at once: each [`Bytes`] is read just-in-time and dropped right after.
///
/// # Semantics
///
/// The data reflects the device state at *first access*, not at construction time. It is
/// therefore only sound for buffers that are not mutated between [`read_lazy`] and the first
/// read. This matches the typical use case of serializing frozen weights.
///
/// [`read_lazy`]: ComputeClient::read_lazy
pub struct LazyDeviceController<R: Runtime> {
    client: ComputeClient<R>,
    descriptor: Arc<CopyDescriptor>,
    /// Host bytes, materialized from the device on first access.
    materialized: Once<Bytes>,
}

impl<R: Runtime> LazyDeviceController<R> {
    pub(super) fn new(client: ComputeClient<R>, descriptor: Arc<CopyDescriptor>) -> Self {
        Self {
            client,
            descriptor,
            materialized: Once::new(),
        }
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

        self.materialized
            .try_call_once(|| -> Result<Bytes, AccessError> {
                let desc = self.descriptor.as_ref();
                // `read_one_tensor_async` consumes the descriptor by value; rebuild one from the
                // shared fields. All clones are cheap (the binding is `Arc`-backed).
                let descriptor = CopyDescriptor::new(
                    desc.handle.clone(),
                    desc.shape.clone(),
                    desc.strides.clone(),
                    desc.elem_size,
                );
                cubecl_common::reader::read_sync(self.client.read_one_tensor_async(descriptor))
                    .map_err(|err| AccessError::Read(format!("{err:?}")))
            })
    }

    /// The host byte length, derived from the descriptor without materializing.
    fn byte_len(&self) -> usize {
        let desc = self.descriptor.as_ref();
        desc.shape.iter().product::<usize>() * desc.elem_size
    }
}

impl<R: Runtime> AllocationController for LazyDeviceController<R> {
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

    // The host byte length, known from the descriptor without materializing.
    fn capacity(&self) -> usize {
        self.byte_len()
    }

    /// A zero-copy (until materialized) device sub-view over the byte range `[start, end)`:
    /// a new lazy controller that, when read, copies only those bytes off the device.
    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        // Once materialized to host there is no device sub-view to take (and the host
        // controller doesn't support views either).
        if self.materialized.get().is_some() {
            return None;
        }
        if start > end || end > self.byte_len() {
            return None;
        }

        let desc = self.descriptor.as_ref();
        // A byte sub-range only maps to a contiguous device region when the source is
        // contiguous (a strided/padded tensor has gaps the byte offsets can't express).
        if !has_contiguous_row_major_strides(&desc.shape, &desc.strides) {
            return None;
        }

        // Narrow the binding to the byte window `[start, end)`. Offsets are byte-level and
        // `offset_end` is trimmed-from-the-end (see `Binding::size_in_used`).
        let base = desc.handle.offset_start.unwrap_or(0);
        let size = desc.handle.size;
        let mut binding = desc.handle.clone();
        binding.offset_start = Some(base + start as u64);
        binding.offset_end = Some(size - (base + end as u64));

        // Flat byte descriptor over the sub-range; materializing reads only these bytes.
        let descriptor = CopyDescriptor::new(binding, [end - start].into(), [1].into(), 1);

        Some(Box::new(LazyDeviceController::new(
            self.client.clone(),
            Arc::new(descriptor),
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
