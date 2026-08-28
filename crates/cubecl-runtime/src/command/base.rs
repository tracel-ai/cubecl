//! One unit of work against the device.
//!
//! Every operation a backend server exposes that touches memory or launches a
//! kernel goes through a [`Command`]: it pairs the context holding the
//! compiled kernels with the streams the operation was resolved against, and
//! resolving is what orders the current stream behind whichever streams own
//! the buffers it was handed.
//!
//! Everything here is the same whichever driver is underneath — the allocation
//! and reclaim policy, when the drop queue may be flushed, what a copy stages.
//! The four calls that are not are [`Driver`](super::Driver)'s.

use super::{CopyLayout, DeviceResource, DeviceStream, Driver, Staging};
use crate::id::KernelId;
use crate::memory_management::drop_queue::Fence;
use crate::memory_management::{
    InstallMemoryPoolsError, ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration,
    MemoryHandle, MemoryReport, MemoryUsage,
};
use crate::server::{BufferBinding, CopyDescriptor, Handle, IoError, LaunchError, ServerError};
use crate::stream::ResolvedStreams;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;

/// One unit of work against the device: the context that holds its compiled
/// kernels, and the streams it was resolved against.
///
/// Built per operation rather than held, because resolving is what orders the
/// current stream behind whichever streams own the buffers it was given.
pub struct Command<'a, D: Driver> {
    ctx: &'a mut D::Context,
    streams: ResolvedStreams<'a, D::Backend>,
}

impl<'a, D: Driver> Command<'a, D> {
    /// A command against `ctx` over the streams `streams` resolved.
    pub fn new(ctx: &'a mut D::Context, streams: ResolvedStreams<'a, D::Backend>) -> Self {
        Self { ctx, streams }
    }

    /// The stream this command is issued on.
    ///
    /// The one part of the resolution a backend reaches for directly: the
    /// driver calls take a stream, and this is the one they take.
    pub fn stream(&mut self) -> &mut D::Stream {
        self.streams.current()
    }

    /// The device allocation `binding` names, resolved on the stream that
    /// created it rather than the current one.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageHandleNotFound`] when the binding names no live allocation.
    pub fn resource(&mut self, binding: BufferBinding) -> Result<DeviceResource<D>, IoError> {
        self.streams
            .get(&binding.stream)
            .device_memory()
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// The current stream's device memory usage.
    pub fn memory_usage(&mut self) -> MemoryUsage {
        self.streams.current().device_memory().memory_usage()
    }

    /// Structured per-pool report of the current stream's device memory.
    pub fn memory_report(&mut self) -> MemoryReport {
        self.streams.current().device_memory().memory_report()
    }

    /// Release everything the current stream is holding that nothing still
    /// needs.
    pub fn memory_cleanup(&mut self) {
        let stream = self.streams.current();
        // Deferred frees sit in the drop queue until a fenced flush, so an
        // explicit cleanup must drain it first or the pools still see those
        // slices as live. Skipped mid-capture: a host sync aborts the capture,
        // and the capture path drains the queue itself. The cleanups below stay
        // safe mid-capture: `cleanup` defers all frees while a capture is
        // active.
        if !stream.capturing().is_recording() {
            let signal = stream.signal();
            stream.drop_queue().drain(|| D::Stream::fence(signal));
            // The info cache's buffers are live slices in the dynamic pools;
            // an explicit cleanup exists to leave those pools empty (e.g. for
            // a rebuild sized to the next workload), so every entry not pinned
            // by a live graph goes too. Skipped while recording for the same
            // reason the drain is: an entry the recording has not touched yet
            // would come back as a fresh allocation inside the capture window,
            // which is illegal.
            stream.info_cache().clear_unpinned();
        }
        let (stream, failures) = self.streams.current_and_failures();
        stream.device_memory().cleanup(true, failures);
        stream.host_memory().cleanup(true, failures);
    }

    /// Flush the current stream's drop queue, freeing what the device is
    /// known to be done with.
    ///
    /// Deferred while the stream records a graph — the flush records a fence
    /// on the capturing stream, which corrupts the recording — and the window
    /// drains the queue itself when it closes. The rule lives here, on the one
    /// path a server has to the queue, so no call site can rebuild the flush
    /// without the guard.
    pub fn flush_drops(&mut self) {
        let stream = self.streams.current();
        if stream.capturing().is_recording() {
            return;
        }
        let signal = stream.signal();
        stream.drop_queue().flush(|| D::Stream::fence(signal));
    }

    /// Set the [`MemoryAllocationMode`] for the current stream.
    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.streams.current().device_memory().mode(mode)
    }

    /// Rebuild the current stream's device pools with a new layout, keeping
    /// the old one when something is still live in them.
    ///
    /// # Errors
    ///
    /// [`InstallMemoryPoolsError::PoolsInUse`] when the rebuild was refused.
    pub fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        props: &MemoryDeviceProperties,
    ) -> Result<(), InstallMemoryPoolsError> {
        let (stream, failures) = self.streams.current_and_failures();
        stream
            .device_memory()
            .install_pools(config, props, failures)
    }

    /// Allocate `size` bytes of device memory on the current stream.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no device could ever fit it, and
    /// whatever the allocator reports when a reclaim-and-retry still cannot.
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        let (stream, failures) = self.streams.current_and_failures();
        match stream.device_memory().reserve(size, failures) {
            Ok(handle) => Ok(handle),
            Err(err) if !err.may_succeed_after_reclaim() => Err(err),
            // Reclaim this stream's memory and retry once; only a failure after
            // that is reported. Without the retry a transient peak becomes a
            // never-initialized handle whose every downstream use fails.
            Err(err) => {
                log::warn!("device allocation of {size} B failed ({err}); reclaiming and retrying");
                self.memory_cleanup();
                let (stream, failures) = self.streams.current_and_failures();
                stream.device_memory().reserve(size, failures)
            }
        }
    }

    /// The current stream's cursor.
    pub fn cursor(&self) -> u64 {
        self.streams.cursor
    }

    /// Allocate `size` bytes of device memory and a handle naming it.
    ///
    /// # Errors
    ///
    /// Whatever the allocation or the bind reports.
    pub fn empty(&mut self, size: u64) -> Result<Handle, IoError> {
        let handle = Handle::new(self.streams.current, size);
        let reserved = self.reserve(size)?;
        self.bind(reserved, handle.memory.clone())?;

        Ok(handle)
    }

    /// Give `reserved`'s storage to `new`, so handles issued against `new`
    /// resolve to it.
    ///
    /// # Errors
    ///
    /// [`IoError`] when the reservation has no initialized storage to give.
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
    ) -> Result<(), IoError> {
        let cursor = self.cursor();
        let (stream, failures) = self.streams.current_and_failures();
        stream.device_memory().bind(reserved, new, cursor, failures)
    }

    /// `size` bytes of host memory, pinned when the pool can serve it.
    ///
    /// Pinned pages transfer by DMA without a bounce, but they are scarce, so
    /// an exhausted pool falls back to the heap rather than failing: this
    /// always answers with a buffer of the size asked for.
    pub fn reserve_cpu(&mut self, size: usize, origin: Option<StreamId>) -> Bytes {
        self.reserve_pinned(size, origin)
            .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]))
    }

    /// `size` bytes of pinned host memory, or `None` when the pool cannot
    /// serve it.
    fn reserve_pinned(&mut self, size: usize, origin: Option<StreamId>) -> Option<Bytes> {
        let (stream, failures) = match origin {
            Some(id) => self.streams.get_and_failures(&id),
            None => self.streams.current_and_failures(),
        };
        let handle = stream.host_memory().reserve(size as u64, failures).ok()?;

        let binding = MemoryHandle::binding(handle);
        let resource = stream
            .host_memory()
            .get_resource(binding.clone(), None, None)
            .ok()?;

        // SAFETY: the binding has initialized memory for at least `size` bytes,
        // and `resource` is what the manager just resolved it to.
        Some(unsafe { D::pinned_bytes(binding, resource, size) })
    }

    /// Copy each descriptor's device memory back to the host, resolving once
    /// the copies have landed.
    ///
    /// The copies are enqueued before the future is returned; awaiting it
    /// waits on the fence that follows them.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout the driver cannot copy,
    /// and whatever the fence reports when the stream itself failed.
    pub fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send + use<D> {
        let held = descriptors
            .iter()
            .map(|descriptor| descriptor.handle.clone())
            .collect::<Vec<_>>();
        let result = self.copies_to_bytes(descriptors);
        let fence = D::Stream::fence(self.streams.current().signal());

        async move {
            let synced = fence.wait();
            // The bindings kept the source allocations alive across the copies;
            // the fence above is what says they are done being read.
            core::mem::drop(held);

            synced?;
            result.map_err(Into::into)
        }
    }

    /// Copy each descriptor's device memory into a fresh host buffer.
    fn copies_to_bytes(&mut self, descriptors: Vec<CopyDescriptor>) -> Result<Vec<Bytes>, IoError> {
        let mut result = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            match self.copy_to_bytes(descriptor, None) {
                Ok(bytes) => result.push(bytes),
                Err(err) => {
                    // The buffers collected so far are the destinations of
                    // copies already enqueued: dropping them hands their
                    // pinned slices back to a pool whose reuse is gated on
                    // the refcount alone, while the device is still writing
                    // them. The fence `read_async` records to cover exactly
                    // this does not exist yet on the error path, so record
                    // one here and wait it out before the partial set drops.
                    if !result.is_empty() {
                        D::Stream::fence(self.streams.current().signal()).sync();
                    }
                    return Err(err);
                }
            }
        }

        Ok(result)
    }

    /// Copy one descriptor's device memory into a fresh host buffer.
    fn copy_to_bytes(
        &mut self,
        descriptor: CopyDescriptor,
        stream_id: Option<StreamId>,
    ) -> Result<Bytes, IoError> {
        let num_bytes = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut bytes = self.reserve_cpu(num_bytes, stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Enqueue a copy of `descriptor`'s device memory into `bytes`.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major, [`IoError::StorageHandleNotFound`] for a binding that names no live
    /// allocation, and the driver's refusal to copy.
    pub fn write_to_cpu(
        &mut self,
        descriptor: CopyDescriptor,
        bytes: &mut Bytes,
        stream_id: Option<StreamId>,
    ) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;
        let layout = CopyLayout::of(&shape, &strides, elem_size)?;

        // Nothing to copy for an empty tensor, and `bytes` has no real backing
        // for the driver to write into — a dangling zero-size buffer.
        if bytes.is_empty() {
            return Ok(());
        }

        let resource = self.resource(binding)?;
        let stream = match stream_id {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };

        // SAFETY: `resource` is a live device allocation the manager just
        // resolved, `bytes` was sized for this copy, and the caller awaits the
        // fence `read_async` records before reading it back.
        unsafe { D::copy_to_host(&resource, &layout, bytes, stream) }
    }

    /// Enqueue a copy of `data` into the device memory `descriptor` names.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major, [`IoError::StorageHandleNotFound`] for a binding that names no live
    /// allocation, and the driver's refusal to copy.
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;
        let layout = CopyLayout::of(&shape, &strides, elem_size)?;

        let resource = self.resource(binding)?;
        let size = data.len();

        // An empty tensor (a zero dim in its shape) has nothing to copy. Bail
        // before staging: the zero-size staging buffer has no real backing (a
        // dangling pointer), and a 2D copy would still transfer `width_bytes`
        // from it when only the leading dims are zero.
        if size == 0 {
            return Ok(());
        }

        let staging = Staging::of(size, data.property());

        let data = match staging.through_pinned {
            true => {
                // Pinned staging is a DMA optimization, not a requirement, so
                // an exhausted pinned pool falls back to a plain heap buffer
                // rather than failing the write — the same answer `reserve_cpu`
                // gives for the same condition. File-backed data still lands in
                // real memory before the driver reads it asynchronously, which
                // is the half of the staging that is mandatory.
                let mut buffer = self
                    .reserve_pinned(size, None)
                    .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]));
                data.copy_into(&mut buffer);
                buffer
            }
            false => data,
        };

        let current = self.streams.current();

        // SAFETY: `resource` is a live device allocation, `data` is a valid
        // host buffer, and either the drop queue or the capture window below
        // keeps it alive for as long as the device reads it.
        unsafe { D::copy_to_device(&resource, &layout, &data, current)? };

        if current.capturing().is_recording() {
            // A copy recorded into a graph is not executed now but re-read on
            // every replay: the node keeps the raw host pointer, so the bytes
            // ride the window onto the graph rather than the drop queue —
            // which frees them when the window closes, exactly when the graph
            // starts needing them.
            current.capturing().retain_host(data);
        } else {
            current.drop_queue().push(data);
            if staging.flush_after || current.drop_queue().should_flush() {
                let signal = current.signal();
                current.drop_queue().flush(|| D::Stream::fence(signal));
            }
        }

        Ok(())
    }

    /// Allocate device memory for `data` and enqueue the copy into it.
    ///
    /// # Errors
    ///
    /// Whatever the allocation or the copy reports.
    pub fn create_with_data(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        let mut staging =
            self.reserve_pinned(data.len(), None)
                .ok_or_else(|| IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "Unable to reserve pinned memory".into(),
                })?;

        staging.copy_from_slice(data);

        let handle = self.empty(staging.len() as u64)?;

        self.write_to_gpu(
            CopyDescriptor {
                handle: handle.clone().binding(),
                shape: [data.len()].into(),
                strides: [1].into(),
                elem_size: 1,
            },
            staging,
        )?;

        Ok(handle)
    }

    /// Wait for everything already enqueued on the current stream to finish.
    ///
    /// # Errors
    ///
    /// The fault the barrier reveals, when the stream itself failed.
    pub fn sync(&mut self) -> DynFut<Result<(), ServerError>> {
        let fence = D::Stream::fence(self.streams.current().signal());

        Box::pin(async move { fence.wait() })
    }

    /// Enqueue an already-compiled kernel on the current stream.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the launch, returned whether or not a
    /// profile is open. An open profile is not a reason to hold the failure
    /// here: the caller's write scope is what claims the buffers the launch
    /// never wrote, and the caller invalidates every open profile on the same
    /// path, so keeping it would lose the claim and duplicate the report.
    pub fn kernel(
        &mut self,
        kernel: KernelId,
        count: (u32, u32, u32),
        args: &mut D::LaunchArgs,
    ) -> Result<(), LaunchError> {
        let stream = self.streams.current();
        let result = D::launch(self.ctx, stream, kernel, count, args);

        // A fenced flush during capture would abort it; defer until the capture
        // ends, when the deferred staging buffers are reclaimed.
        if !stream.capturing().is_recording() && stream.drop_queue().should_flush() {
            let signal = stream.signal();
            stream.drop_queue().flush(|| D::Stream::fence(signal));
        }

        result
    }
}
