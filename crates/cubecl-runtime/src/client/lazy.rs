//! Allocation controller that lazily reads a device resource into host memory.

use super::ComputeClient;
use crate::runtime::Runtime;
use crate::server::CopyDescriptor;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use cubecl_common::bytes::{AllocationController, AllocationProperty, Bytes};
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
    fn ensure_init(&self) -> &Bytes {
        self.materialized.call_once(|| {
            let desc = self.descriptor.as_ref();
            // `read_one_unchecked_tensor` consumes the descriptor by value; rebuild one from the
            // shared fields. All clones are cheap (the binding is `Arc`-backed).
            let descriptor = CopyDescriptor::new(
                desc.handle.clone(),
                desc.shape.clone(),
                desc.strides.clone(),
                desc.elem_size,
            );
            self.client.read_one_unchecked_tensor(descriptor)
        })
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

    fn memory(&self) -> &[MaybeUninit<u8>] {
        let bytes = self.ensure_init();
        let slice: &[u8] = bytes;
        // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` share a layout, and every byte read from the
        // device is initialized.
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.ensure_init();

        // SAFETY: `ensure_init` guarantees the cell is initialized, and `&mut self` is exclusive.
        let bytes = self
            .materialized
            .get_mut()
            .expect("materialized must be set after init");
        let slice: &mut [u8] = bytes;
        // SAFETY: same layout as above; every byte is initialized.
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        let bytes = self.ensure_init();
        let len = buf.len().min(bytes.len());
        buf[..len].copy_from_slice(&bytes[..len]);
    }
}
