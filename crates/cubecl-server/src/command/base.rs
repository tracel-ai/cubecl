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
//! The calls that are not are [`Driver`](super::Driver)'s.

use super::stream_relocation_copier::StreamRelocationCopier;
use super::{CopyLayout, DeviceResource, DeviceStream, Driver, Staging};
use crate::id::KernelId;
use crate::memory_management::drop_queue::DeviceWaiter;
use crate::memory_management::relocation::{
    DeviceRelocator, RelocationNeed, RelocationReason, RelocationScope,
};
use crate::memory_management::{
    AllocationKind, ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode, PageGuard,
    PageUpdate, StreamMemoryReport,
};
use crate::memory_management::{Cleanup, release_or_leak};
use crate::server::{
    BufferBinding, CopyDescriptor, Handle, IoError, LaunchError, PeerBinding, ServerError,
};
use crate::storage::ManagedResource;
use crate::stream::ResolvedStreams;
use crate::stream::{Placement, TouchedPage};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::{bytes::Bytes, device::ServiceId};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;

/// One unit of work against the device: the context that holds its compiled
/// kernels, and the streams it was resolved against.
///
/// Built per operation rather than held, because resolving is what orders the
/// current stream behind whichever streams own the buffers it was given.
pub struct Command<'a, D: Driver> {
    ctx: &'a mut D::Context,
    streams: ResolvedStreams<'a, D::Backend>,
    /// The service issuing the command: what the handles it allocates are
    /// stamped with.
    service: ServiceId,
}

impl<'a, D: Driver> Command<'a, D> {
    /// A command against `ctx` over the streams `streams` resolved.
    pub fn new(
        ctx: &'a mut D::Context,
        streams: ResolvedStreams<'a, D::Backend>,
        service: ServiceId,
    ) -> Self {
        Self {
            ctx,
            streams,
            service,
        }
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
    /// While the current stream records a graph, the page the allocation
    /// sits on is remembered: the graph replays against the address resolved
    /// here, so it guards that page once it seals.
    ///
    /// # Errors
    ///
    /// [`IoError::BindingOutOfBounds`] when the binding's offsets reach past
    /// the end of its memory.
    pub fn resource(&mut self, binding: BufferBinding) -> Result<DeviceResource<D>, IoError> {
        self.streams
            .current()
            .capturing()
            .touch(binding.stream, binding.memory.descriptor().location());
        self.streams
            .get(&binding.stream)
            .device_memory()
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// The device allocation `binding` names, for a caller outside the server
    /// that keeps its raw address: it holds the allocation's binding for as
    /// long as it lives, with the Guarantees of
    /// [`ManagedResource`](crate::storage::ManagedResource), a graph capture
    /// moving what it keeps apart included.
    ///
    /// # Errors
    ///
    /// [`IoError::BindingOutOfBounds`] when the binding's offsets reach past
    /// the end of its memory.
    pub fn managed_resource(
        &mut self,
        binding: BufferBinding,
    ) -> Result<ManagedResource<DeviceResource<D>>, IoError> {
        self.streams
            .get(&binding.stream)
            .device_memory()
            .managed_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// A guard on every page in `touched`, each looked up in the memory of the
    /// stream that owns it.
    pub fn guard_pages(&mut self, touched: &[TouchedPage]) -> Vec<PageGuard> {
        touched
            .iter()
            .filter_map(|page| {
                self.streams
                    .get(&page.stream)
                    .device_memory()
                    .guard(page.location)
            })
            .collect()
    }

    /// Everything the current stream's device memory holds, pool by pool.
    pub fn memory_report(&mut self) -> StreamMemoryReport {
        StreamMemoryReport {
            stream: self.streams.current,
            pools: self.streams.current().device_memory().memory_report(),
            auxiliary: Vec::new(),
        }
    }

    /// Release everything the current stream is holding that nothing still
    /// needs.
    ///
    /// # Errors
    ///
    /// Refused while any stream records a graph: releasing memory waits on
    /// the device, and a wait on a stream that records aborts its capture.
    pub fn memory_cleanup(&mut self) -> Result<(), ServerError> {
        self.refuse_while_recording()?;
        let stream = self.streams.current();
        // Deferred frees sit in the drop queue until a fenced flush, so an
        // explicit cleanup must drain it first or the pools still see those
        // slices as live.
        let signal = stream.signal();
        stream.drop_queue().drain(|| D::Stream::fence(signal));
        // The info cache's buffers are live slices in the dynamic pools; an
        // explicit cleanup exists to leave those pools empty, so every entry
        // not pinned by a live graph goes too.
        stream.info_cache().clear_unpinned();

        DeviceRelocator::reclaim(self)?;
        let (stream, failures) = self.streams.current_and_failures();
        stream.host_memory().cleanup(Cleanup::Explicit, failures);
        Ok(())
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

    /// Allocate `size` bytes of device memory on the current stream.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no device could ever fit it, and
    /// whatever the allocator reports when a reclaim-and-retry still cannot.
    fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        let current = self.streams.current;
        let kind = self.streams.current().capturing().allocation_kind(current);
        self.reserve_as(size, kind)
    }

    /// [`reserve`](Self::reserve), for an allocation of `kind`.
    ///
    /// # Errors
    ///
    /// As [`reserve`](Self::reserve).
    fn reserve_as(
        &mut self,
        size: u64,
        kind: AllocationKind,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let update = self.streams.current().capturing().page_update();
        if update == PageUpdate::Allow {
            self.relocate_when_wanted();
        }

        let (stream, failures) = self.streams.current_and_failures();
        match stream
            .device_memory()
            .reserve_as(size, update, kind, failures)
        {
            Ok(handle) => Ok(handle),
            // The recording now misses whatever this memory was for.
            Err(err @ IoError::PageUpdateForbidden { .. }) => {
                stream.capturing().fail(err.clone().into());
                Err(err)
            }
            Err(err) if update != PageUpdate::Allow || !err.may_succeed_after_reclaim() => Err(err),
            // Reclaim and retry once; only a failure after that is reported.
            // Without the retry a transient peak becomes a never-initialized
            // handle whose every downstream use fails.
            // Retried where it was asked for: a capture's output landing on
            // the shared room, or a temporary taking a fresh page, would be
            // placed otherwise than the plan the recording follows.
            Err(err) => {
                log::warn!("device allocation of {size} B failed ({err}); reclaiming and retrying");
                self.make_room(&err, kind);
                let (stream, failures) = self.streams.current_and_failures();
                stream
                    .device_memory()
                    .reserve_as(size, PageUpdate::Allow, kind, failures)
            }
        }
    }

    /// Get back what the reservation that failed with `err` needs: a slot for
    /// a new page size when the arena an allocation of `kind` goes to is full,
    /// memory otherwise.
    fn make_room(&mut self, err: &IoError, kind: AllocationKind) {
        match (err, kind) {
            // Only a relocation that may allocate empties the outdated pools
            // whatever room the current pages have.
            (
                IoError::PageSizesExhausted { .. },
                AllocationKind::Eager | AllocationKind::GraphIntermediate,
            ) => DeviceRelocator::relocate(self, RelocationReason::ArenaFull),
            // Nothing frees a slot of the separate arena: it is never
            // relocated, since its graphs pin what it holds, and every
            // reservation already drained its outdated pools that emptied. The
            // retry reports the exhaustion, rather than placing the output on
            // the shared room.
            (IoError::PageSizesExhausted { .. }, AllocationKind::GraphOutput) => {}
            // No stream records (the caller checked), which is the only
            // refusal a cleanup has.
            _ => {
                let _ = self.memory_cleanup();
            }
        }
    }

    /// Give `memory` `size` bytes of device memory on the current stream.
    ///
    /// `memory` is the handle the task that asked for it owns, and it ends in the pool, moved
    /// rather than cloned: from the ask to the pool, the buffer has exactly one handle besides
    /// its users', so what [`can_mut`](ManagedMemoryHandle::can_mut) answers them never depends
    /// on how far the device thread has got.
    ///
    /// # Errors
    ///
    /// Returns an [`IoError`] when the memory cannot be reserved or bound, e.g. when the device
    /// is out of memory. The caller records it on `memory`, which has no binding to carry it:
    /// it takes `memory`'s [binding](ManagedMemoryHandle::to_binding) before handing it over.
    pub fn initialize_memory(
        &mut self,
        memory: ManagedMemoryHandle,
        size: u64,
    ) -> Result<(), IoError> {
        match self.allocate(size, memory) {
            Ok(()) => Ok(()),
            // The recording already failed on it, and `stop_capture` reports
            // that: the handle stays unbound, and whatever uses it belongs to
            // a recording that will not seal.
            Err(IoError::PageUpdateForbidden { .. }) => Ok(()),
            Err(err) => Err(err),
        }
    }

    /// Give `memory` `size` bytes, where the capture under way places it: what the planning run
    /// found to be an output off the room the stream's graphs pinned, anything else where the
    /// pools put it. Every device allocation of a command goes through here, so a capture sees
    /// each one, in order.
    ///
    /// # Errors
    ///
    /// Whatever the reservation or the bind reports. `memory` is dropped then.
    pub fn allocate(&mut self, size: u64, memory: ManagedMemoryHandle) -> Result<(), IoError> {
        let current = self.streams.current;
        let placement = self.streams.current().capturing().place(current, size);
        self.allocate_at(size, memory, placement)
    }

    /// [`allocate`](Self::allocate) for what a graph keeps while a capture is under way: room
    /// apart from what the stream's graphs share, whatever the plan says.
    ///
    /// Left out of the run's sequence: its placement never depends on the plan, and a launch's
    /// metadata is built only where the cache misses, which the planning runs and the recording
    /// need not agree on — a scalar that changes between runs misses once more.
    fn allocate_separate(&mut self, size: u64, memory: ManagedMemoryHandle) -> Result<(), IoError> {
        let current = self.streams.current;
        let reserved = match self.streams.current().capturing().is_owned_by(current) {
            true => self.reserve_as(size, AllocationKind::GraphOutput)?,
            false => self.reserve(size)?,
        };
        self.bind(reserved, memory);
        Ok(())
    }

    fn allocate_at(
        &mut self,
        size: u64,
        memory: ManagedMemoryHandle,
        placement: Placement,
    ) -> Result<(), IoError> {
        let reserved = match placement {
            Placement::Output => self.reserve_as(size, AllocationKind::GraphOutput)?,
            Placement::Pools => self.reserve(size)?,
        };
        // Taken before the handle moves into the pool. A recorded output's binding outlives this
        // command: the graph it seals into writes the buffer on every replay, in stream order.
        let binding = memory.to_binding();
        self.bind(reserved, memory);
        let current = self.streams.current;
        self.streams
            .current()
            .capturing()
            .allocated(current, size, binding, placement);
        Ok(())
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
        let handle = Handle::new(self.service, self.streams.current, size);
        self.allocate(size, handle.memory.clone())?;

        Ok(handle)
    }

    /// Give `reserved`'s storage to `new`, so handles issued against `new`
    /// resolve to it.
    fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        let cursor = self.cursor();
        let (stream, failures) = self.streams.current_and_failures();
        stream.device_memory().bind(reserved, new, cursor, failures);
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
        // A stream recording a graph stages from what the pool holds, and
        // falls back to the heap rather than allocate.
        let update = stream.capturing().page_update();
        let handle = stream
            .host_memory()
            .reserve(size as u64, update, failures)
            .ok()?;

        let binding = handle.binding();
        stream.capturing().prime(&binding);
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
        let copies_any = descriptors
            .iter()
            .any(|descriptor| num_bytes(descriptor) != 0);
        let result = self.copies_to_bytes(descriptors);
        let fence = D::Stream::fence(self.streams.current().signal());
        let copies = match copies_any && result.is_ok() {
            false => InFlight::unguarded(fence, (held, result)),
            true => InFlight::new(fence, (held, result)),
        };

        async move {
            let (held, result) = copies.wait()?;
            core::mem::drop(held);

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
                    if !result.is_empty()
                        && let Err(waited) =
                            D::Stream::fence(self.streams.current().signal()).wait()
                    {
                        crate::memory_management::release_or_leak(result, &waited);
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
        let mut bytes = self.reserve_cpu(num_bytes(&descriptor), stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Enqueue a copy of `descriptor`'s device memory into `bytes`.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major, [`IoError::BindingOutOfBounds`] for a binding whose offsets reach
    /// past the end of its memory, and the driver's refusal to copy.
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
        // Nothing to copy for an empty tensor, and `bytes` has no real backing
        // for the driver to write into — a dangling zero-size buffer.
        // Its strides may contain zeros, so skip copy-layout validation too.
        if bytes.is_empty() {
            return Ok(());
        }

        let layout = CopyLayout::of(&shape, &strides, elem_size)?;
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
    /// row-major, [`IoError::BindingOutOfBounds`] for a binding whose offsets reach
    /// past the end of its memory, and the driver's refusal to copy.
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;
        let size = data.len();

        // An empty tensor (a zero dim in its shape) has nothing to copy. Bail
        // before validating its potentially zero strides or staging: the zero-size
        // staging buffer has no real backing (a dangling pointer), and a 2D copy
        // would still transfer `width_bytes` from it when only the leading dims are zero.
        if size == 0 {
            return Ok(());
        }

        let layout = CopyLayout::of(&shape, &strides, elem_size)?;
        let resource = self.resource(binding)?;
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
        // keeps it alive until the fence after the copy signals. A stream torn
        // down with the copy in flight drops the queue, which leaks the bytes
        // rather than freeing them under the copy.
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
        self.create_on(data, Room::Planned)
    }

    /// [`create_with_data`](Self::create_with_data) for a launch's metadata: while a capture is
    /// under way, the buffer may be kept for the graph's life (the info cache keeps everything a
    /// capture builds), so it goes to room no graph's replay writes, however large it is.
    ///
    /// # Errors
    ///
    /// As [`create_with_data`](Self::create_with_data).
    pub fn create_metadata(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        self.create_on(data, Room::Separate)
    }

    fn create_on(&mut self, data: &[u8], room: Room) -> Result<Handle, IoError> {
        let mut staging =
            self.reserve_pinned(data.len(), None)
                .ok_or_else(|| IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "Unable to reserve pinned memory".into(),
                })?;

        staging.copy_from_slice(data);

        let size = staging.len() as u64;
        let handle = Handle::new(self.service, self.streams.current, size);
        match room {
            Room::Planned => self.allocate(size, handle.memory.clone())?,
            Room::Separate => self.allocate_separate(size, handle.memory.clone())?,
        }

        self.write_to_gpu(
            CopyDescriptor {
                handle: handle.to_binding(),
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
        // The memory of other devices this stream binds stays pinned on its
        // owner until a flush releases it, so a sync, after which nothing here
        // may be running, releases it rather than leave it pinned while the
        // stream idles.
        let stream = self.streams.current();
        if !stream.capturing().is_recording() && stream.drop_queue().holds_peers() {
            let signal = stream.signal();
            stream.drop_queue().drain(|| D::Stream::fence(signal));
        }
        let fence = D::Stream::fence(self.streams.current().signal());

        Box::pin(async move { fence.wait() })
    }

    /// Keep `peers`, memory of other devices the kernel just enqueued binds,
    /// allocated and in place there until the current stream is past that
    /// kernel. Their owners order nothing against this stream, so this is what
    /// tells them the memory is still in use. While the stream records a
    /// graph, the graph holds them instead, for as long as it can be replayed.
    pub fn hold_peers(&mut self, peers: impl IntoIterator<Item = PeerBinding>) {
        let stream = self.streams.current();
        let recording = stream.capturing().is_recording();
        for peer in peers {
            match recording {
                true => stream.capturing().retain_peer(peer.into_binding()),
                false => stream.drop_queue().hold_peer(peer.into_binding()),
            }
        }
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
        id: &KernelId,
        kernel: &D::Loaded,
        count: (u32, u32, u32),
        args: &mut D::LaunchArgs,
    ) -> Result<(), LaunchError> {
        let stream = self.streams.current();
        let result = D::launch(self.ctx, stream, id, kernel, count, args);

        // A fenced flush during capture would abort it; defer until the capture
        // ends, when the deferred staging buffers are reclaimed.
        if !stream.capturing().is_recording() && stream.drop_queue().should_flush() {
            let signal = stream.signal();
            stream.drop_queue().flush(|| D::Stream::fence(signal));
        }

        result
    }
}

impl<D: Driver> DeviceRelocator for Command<'_, D> {
    fn relocation_need(&mut self, scope: RelocationScope) -> RelocationNeed {
        if self.streams.current().capturing().device().any_recording() {
            return RelocationNeed::Recording;
        }
        match scope {
            RelocationScope::Stream => self.stream_need(),
            RelocationScope::Device => RelocationNeed::of_device(
                self.streams
                    .all()
                    .map(|stream| stream.device_memory().relocation_need()),
            ),
        }
    }

    fn cleanup_memory(&mut self) {
        let (stream, failures) = self.streams.current_and_failures();
        stream.device_memory().cleanup(Cleanup::Explicit, failures);
    }

    fn relocate_memory(&mut self, scope: RelocationScope, reason: RelocationReason) {
        if self.streams.current().capturing().device().any_recording() {
            return;
        }
        let stream_ids: Vec<_> = match scope {
            RelocationScope::Stream => alloc::vec![self.streams.current],
            RelocationScope::Device => self.streams.stream_ids(),
        };
        for stream_id in stream_ids {
            self.relocate_stream(stream_id, reason);
        }
    }
}

impl<D: Driver> Command<'_, D> {
    /// Move every allocation of `kept` that sits on a page a graph guards, in the current
    /// stream's memory, onto room no graph's replay writes, with the copies queued on that
    /// stream (see [`MemoryManagement::move_apart`](crate::memory_management::MemoryManagement::move_apart)).
    ///
    /// # Errors
    ///
    /// Refused while any stream records, since the copies wait on the whole device, and when
    /// one does not move.
    pub fn move_apart(&mut self, kept: &[ManagedMemoryBinding]) -> Result<(), ServerError> {
        if kept.is_empty() {
            return Ok(());
        }
        self.refuse_while_recording()?;
        let current = self.streams.current;
        let mut copier = StreamRelocationCopier::<D>::new(&mut self.streams, current, self.ctx);
        let (stream, failures) = self.streams.current_and_failures();
        stream
            .device_memory()
            .move_apart(kept, &mut copier, failures)
    }

    /// The current stream's need, resolved against the bytes every stream's
    /// memory holds.
    fn stream_need(&mut self) -> RelocationNeed {
        let need = self.streams.current().device_memory().relocation_need();
        need.resolve(|| {
            self.streams
                .all()
                .map(|stream| stream.device_memory().bytes_allocated())
                .sum()
        })
    }

    /// Move what `stream_id`'s memory holds on outdated pages, with the copies
    /// queued on that stream.
    fn relocate_stream(&mut self, stream_id: StreamId, reason: RelocationReason) {
        let mut copier = StreamRelocationCopier::<D>::new(&mut self.streams, stream_id, self.ctx);
        let (stream, failures) = self.streams.get_and_failures(&stream_id);
        stream
            .device_memory()
            .relocate(&mut copier, reason, failures);
    }
}

/// Where a buffer [`Command::create_on`] creates is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Room {
    /// Where the capture's plan puts it, as [`Command::allocate`] does.
    Planned,
    /// Apart from the room the stream's graphs share, while a capture is under way.
    Separate,
}

/// How many bytes reading `descriptor` copies to the host.
fn num_bytes(descriptor: &CopyDescriptor) -> usize {
    descriptor.shape.iter().product::<usize>() * descriptor.elem_size
}

/// Memory a stream's copies may still be using until `fence` is reached: their
/// device sources and host destinations. Dropped before [`wait`](Self::wait),
/// it waits for the fence itself, so a read abandoned unawaited never hands a
/// buffer back to its pool mid-copy.
struct InFlight<F: DeviceWaiter, M> {
    fence_and_memory: Option<(F, M)>,
    waits_on_drop: bool,
}

impl<F: DeviceWaiter, M> InFlight<F, M> {
    fn new(fence: F, memory: M) -> Self {
        Self {
            fence_and_memory: Some((fence, memory)),
            waits_on_drop: true,
        }
    }

    /// For memory no copy is using, as when nothing was enqueued: dropped, it
    /// does not wait.
    fn unguarded(fence: F, memory: M) -> Self {
        Self {
            fence_and_memory: Some((fence, memory)),
            waits_on_drop: false,
        }
    }

    /// The memory, once the device is done with it.
    ///
    /// # Errors
    ///
    /// The fault the fence reports; the memory is then released or leaked.
    fn wait(mut self) -> Result<M, ServerError> {
        let (fence, memory) = self.fence_and_memory.take().expect("waited once");
        match fence.wait() {
            Ok(()) => Ok(memory),
            Err(err) => {
                release_or_leak(memory, &err);
                Err(err)
            }
        }
    }
}

impl<F: DeviceWaiter, M> Drop for InFlight<F, M> {
    fn drop(&mut self) {
        if let Some((fence, memory)) = self.fence_and_memory.take()
            && self.waits_on_drop
            && let Err(err) = fence.wait()
        {
            release_or_leak(memory, &err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct CountingFence(Arc<AtomicUsize>);

    impl DeviceWaiter for CountingFence {
        fn wait(self) -> Result<(), ServerError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Records, when dropped, how many times the fence had been waited on.
    struct Buffer {
        waits: Arc<AtomicUsize>,
        waits_at_release: Arc<AtomicUsize>,
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            let waits = self.waits.load(Ordering::SeqCst);
            self.waits_at_release.store(waits, Ordering::SeqCst);
        }
    }

    fn copies(guarded: bool) -> (InFlight<CountingFence, Buffer>, Arc<AtomicUsize>) {
        let waits = Arc::new(AtomicUsize::new(0));
        let waits_at_release = Arc::new(AtomicUsize::new(usize::MAX));
        let buffer = Buffer {
            waits: waits.clone(),
            waits_at_release: waits_at_release.clone(),
        };
        let fence = CountingFence(waits);
        let copies = match guarded {
            true => InFlight::new(fence, buffer),
            false => InFlight::unguarded(fence, buffer),
        };
        (copies, waits_at_release)
    }

    #[test]
    fn copies_dropped_unawaited_release_their_memory_after_the_fence() {
        let (copies, waits_at_release) = copies(true);

        drop(copies);

        assert_eq!(waits_at_release.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn memory_no_copy_uses_is_released_without_waiting() {
        let (copies, waits_at_release) = copies(false);

        drop(copies);

        assert_eq!(waits_at_release.load(Ordering::SeqCst), 0);
    }
}
