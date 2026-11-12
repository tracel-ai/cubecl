//! A server's half of collectives: checking a collective's buffers, running it on the
//! collective stream once the compute stream has written them, and leaving its destination
//! carrying any failure, the only report a fire-and-forget submit keeps.
//!
//! The same whichever library is underneath, so a runtime supplies where its parts are and its
//! `ServerCommunication` delegates here.

use super::{
    CollectiveDriver, Collectives, Command, DeviceResource, DeviceStream, Driver, StreamSignal,
};
use crate::device_events::{EventApi, EventProfiler, Mark};
use crate::driver::DriverError;
use crate::server::{
    BufferBinding, CollectiveBuffer, CommunicationId, CommunicationTiming, ComputeWait,
    CopyDescriptor, Handle, ReduceOperation, Reduction, SendOutcome, SendReport, ServerError,
};
use crate::stream::{FailureStore, MultiStream};
use alloc::{format, vec, vec::Vec};
use core::convert::Infallible;
use cubecl_common::device::DeviceId;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{ElemType, FloatKind, UIntKind};

/// What a collective that only moves data names its buffers as, whatever they hold.
const BYTES: ElemType = ElemType::UInt(UIntKind::U8);

/// The size of [`CollectiveServer::connect_element`]: one `f32`.
pub const CONNECT_ELEMENT_SIZE: u64 = 4;

/// A server whose device runs collectives and transfers through a [`CollectiveDriver`].
pub trait CollectiveServer: Sized {
    /// The library collectives run through.
    type Driver: CollectiveDriver;

    /// The device's events, which time a communication when an observer asks.
    type Events: EventApi<Stream = StreamSignal<Self::Driver>>;

    /// This server's device.
    fn device_id(&self) -> DeviceId;

    /// The groups this device has joined.
    fn collectives(&mut self) -> &mut Collectives<Self::Driver>;

    /// The profiler that times this device's kernels, whose marks time its communications too.
    fn profiler(&mut self) -> &mut EventProfiler<Self::Events>;

    /// The compute streams, and the failures their buffers carry.
    fn streams(&mut self) -> &mut MultiStream<<Self::Driver as Driver>::Backend>;

    /// A command on `stream_id` over `handles`, with this device current.
    ///
    /// # Errors
    ///
    /// A stream that cannot be created or aligned with the handles' own.
    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_, Self::Driver>, ServerError>;

    /// The stream collectives run on, apart from the compute streams so a collective never
    /// blocks one.
    fn comm_stream(&self) -> StreamSignal<Self::Driver>;

    /// The stream transfers between devices run on, apart from the collectives', so neither
    /// queues behind the other.
    fn transfer_stream(&self) -> StreamSignal<Self::Driver>;

    /// [`CONNECT_ELEMENT_SIZE`] bytes a communicator's first `all_reduce` runs over, allocated
    /// with the device so that setting a group up never fails to allocate on one device after
    /// joining on the others.
    fn connect_element(&self) -> &DeviceResource<Self::Driver>;

    /// Joins the group over `device_ids` and connects its peers, unless this device already has.
    ///
    /// # Errors
    ///
    /// The driver's refusal to join or connect.
    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        // A group already joined is joined once, so the membership is announced once too.
        if let Some(id) = self.collectives().join(device_ids)? {
            connect(self, &id)?;
        }
        Ok(())
    }

    /// See `ServerCommunication::all_reduce`.
    ///
    /// # Errors
    ///
    /// A refusal, which `dst` carries as well.
    // One argument per thing the part is told; a struct over them would name nothing.
    #[allow(clippy::too_many_arguments)]
    fn all_reduce(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        reduction: Reduction,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
        wait: ComputeWait,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let size = src.size_in_used();
        let checked = CollectiveBuffer::Destination
            .expect_size(&dst, size)
            .and_then(|()| Self::Driver::elements(reduction.dtype, size));
        let group = CommunicationId::from(device_ids);
        settle(
            self,
            &group,
            [src],
            dst,
            checked,
            |server, [src], dst, elements| {
                run(
                    server,
                    stream_id,
                    &group,
                    [src, dst],
                    wait,
                    timing,
                    |comm, [src, dst], stream| {
                        Self::Driver::all_reduce(comm, src, dst, elements, reduction.op, stream)
                    },
                )
            },
        )
    }

    /// See `ServerCommunication::all_gather`.
    ///
    /// # Errors
    ///
    /// A refusal, which `dst` carries as well.
    fn all_gather(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let size = src.size_in_used();
        let checked = CollectiveBuffer::Destination
            .expect_size(&dst, size * device_ids.len() as u64)
            .and_then(|()| Self::Driver::elements(BYTES, size));
        let group = CommunicationId::from(device_ids);
        settle(
            self,
            &group,
            [src],
            dst,
            checked,
            |server, [src], dst, elements| {
                run(
                    server,
                    stream_id,
                    &group,
                    [src, dst],
                    ComputeWait::Now,
                    timing,
                    |comm, [src, dst], stream| {
                        Self::Driver::all_gather(comm, src, dst, elements, stream)
                    },
                )
            },
        )
    }

    /// See `ServerCommunication::reduce_scatter`.
    ///
    /// # Errors
    ///
    /// A refusal, which `dst` carries as well.
    fn reduce_scatter(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        reduction: Reduction,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let size = dst.size_in_used();
        let checked = CollectiveBuffer::Source
            .expect_size(&src, size * device_ids.len() as u64)
            .and_then(|()| Self::Driver::elements(reduction.dtype, size));
        let group = CommunicationId::from(device_ids);
        settle(
            self,
            &group,
            [src],
            dst,
            checked,
            |server, [src], dst, elements| {
                run(
                    server,
                    stream_id,
                    &group,
                    [src, dst],
                    ComputeWait::Now,
                    timing,
                    |comm, [src, dst], stream| {
                        Self::Driver::reduce_scatter(comm, src, dst, elements, reduction.op, stream)
                    },
                )
            },
        )
    }

    /// See `ServerCommunication::broadcast`.
    ///
    /// # Errors
    ///
    /// A refusal, which `dst` carries as well.
    fn broadcast(
        &mut self,
        src: Option<BufferBinding>,
        dst: BufferBinding,
        stream_id: StreamId,
        root: DeviceId,
        device_ids: Vec<DeviceId>,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let size = dst.size_in_used();
        let root_rank = self.collectives().rank(root, &device_ids);
        let group = CommunicationId::from(device_ids);
        match (src, self.device_id() == root) {
            (Some(src), true) => {
                let checked = CollectiveBuffer::Source
                    .expect_size(&src, size)
                    .and(root_rank);
                settle(
                    self,
                    &group,
                    [src],
                    dst,
                    checked,
                    |server, [src], dst, root| {
                        run(
                            server,
                            stream_id,
                            &group,
                            [src, dst],
                            ComputeWait::Now,
                            timing,
                            |comm, [src, dst], stream| {
                                let elements = Self::Driver::elements(BYTES, size)?;
                                Self::Driver::broadcast(
                                    comm,
                                    Some(src),
                                    dst,
                                    elements,
                                    root,
                                    stream,
                                )
                            },
                        )
                    },
                )
            }
            (None, false) => settle(self, &group, [], dst, root_rank, |server, [], dst, root| {
                run(
                    server,
                    stream_id,
                    &group,
                    [dst],
                    ComputeWait::Now,
                    timing,
                    |comm, [dst], stream| {
                        let elements = Self::Driver::elements(BYTES, size)?;
                        Self::Driver::broadcast(comm, None, dst, elements, root, stream)
                    },
                )
            }),
            // Only the root sends, so a source anywhere else, or none there, has nothing to send.
            _ => {
                let refused = Err::<Infallible, _>(ServerError::Generic {
                    reason: format!(
                        "a broadcast from {root:?} takes a source on the root and nowhere else"
                    ),
                    backtrace: BackTrace::capture(),
                });
                settle(
                    self,
                    &group,
                    [],
                    dst,
                    refused,
                    |_, [], _, never| match never {},
                )
            }
        }
    }

    /// See `ServerCommunication::all_to_all`.
    ///
    /// # Errors
    ///
    /// A refusal, which `dst` carries as well.
    fn all_to_all(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let ranks = device_ids.len();
        let size = src.size_in_used();
        let checked = CollectiveBuffer::Destination
            .expect_size(&dst, size)
            .and_then(|()| match size.is_multiple_of(ranks as u64) {
                true => Self::Driver::elements(BYTES, size / ranks as u64),
                false => Err(ServerError::Generic {
                    reason: format!(
                        "an all_to_all over {ranks} ranks splits {size} bytes unevenly"
                    ),
                    backtrace: BackTrace::capture(),
                }),
            });
        let group = CommunicationId::from(device_ids);
        settle(
            self,
            &group,
            [src],
            dst,
            checked,
            |server, [src], dst, chunk| {
                run(
                    server,
                    stream_id,
                    &group,
                    [src, dst],
                    ComputeWait::Now,
                    timing,
                    |comm, [src, dst], stream| {
                        Self::Driver::all_to_all(comm, src, dst, chunk, ranks, stream)
                    },
                )
            },
        )
    }

    /// Makes `stream_id` wait for every collective queued so far.
    ///
    /// # Errors
    ///
    /// A stream that cannot be resolved, and the driver's refusal to order the two.
    fn sync_collective(
        &mut self,
        stream_id: StreamId,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        let marks = CommunicationMarks::new(self, timing);
        let mut command = self.command(stream_id, [].into_iter())?;
        let mut marks = marks.unless_capturing(command.stream());
        let stream = command.stream().signal();
        drop(command);
        marks.reach(stream);
        marks.ended(self.comm_stream(), stream, ComputeWait::Now)
    }

    /// Sends `desc` to `device_id_dst`, whose server receives it with [`recv`](Self::recv).
    ///
    /// # Errors
    ///
    /// A source some failure left unwritten, and the driver's refusal to join or send.
    fn send(
        &mut self,
        desc: CopyDescriptor,
        stream_id: StreamId,
        device_id_dst: DeviceId,
        report: SendReport,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        post_send(self, desc, stream_id, device_id_dst, &report, timing)
            .inspect_err(|refusal| report.refused(refusal.clone()))
    }

    /// Receives into `handle` what `device_id_src` sends with [`send`](Self::send).
    ///
    /// # Errors
    ///
    /// A refusal, which `handle` carries as well.
    fn recv(
        &mut self,
        handle: Handle,
        stream_id: StreamId,
        device_id_src: DeviceId,
        outcome: SendOutcome,
        timing: Option<CommunicationTiming>,
    ) -> Result<(), ServerError> {
        // Staged before the handle is consumed below.
        let destination = handle.clone().binding();

        // Every failure from here leaves the destination holding whatever it
        // held, so a read of it has to fail on that rather than take those
        // bytes for a result.
        match receive(self, handle, stream_id, device_id_src, outcome, timing) {
            Ok(()) => {
                // The data is on its way, so an earlier failure that left the
                // destination stale has nothing left to say about it.
                self.streams().written([&destination].into_iter());
                Ok(())
            }
            Err(error) => {
                self.streams()
                    .taint(error.clone(), [&destination].into_iter());
                Err(error)
            }
        }
    }
}

/// Runs one single-element `all_reduce` over the group `id`, so the driver connects its peers
/// now: NCCL and RCCL connect them on a communicator's first operation, which waits on the host
/// for every peer. A single element connects the algorithm they pick, the ring on a single node.
fn connect<S: CollectiveServer>(server: &mut S, id: &CommunicationId) -> Result<(), ServerError> {
    let comm = *server.collectives().get(id)?;
    let element = server.connect_element();
    let elements = S::Driver::elements(ElemType::Float(FloatKind::F32), CONNECT_ELEMENT_SIZE)?;
    S::Driver::all_reduce(
        &comm,
        element,
        element,
        elements,
        ReduceOperation::Sum,
        server.comm_stream(),
    )
}

/// Runs `collective` over `group`, which reads `reads` and writes `dst`, under the
/// tainted-buffer rules: a broken group, a tainted read, a failed `checked` or any later
/// refusal leaves `dst` carrying the failure and breaks the group.
fn settle<S: CollectiveServer, const N: usize, Plan>(
    server: &mut S,
    group: &CommunicationId,
    reads: [BufferBinding; N],
    dst: BufferBinding,
    checked: Result<Plan, ServerError>,
    collective: impl FnOnce(&mut S, [BufferBinding; N], BufferBinding, Plan) -> Result<(), ServerError>,
) -> Result<(), ServerError> {
    let destination = dst.clone();
    let ran = server
        .collectives()
        .ensure_intact(group)
        .and_then(|()| server.streams().ensure_written(reads.iter()))
        .and(checked)
        .and_then(|plan| collective(server, reads, dst, plan));
    match ran {
        Ok(()) => {
            // The result is on its way, so an earlier failure that left the
            // destination stale has nothing left to say about it.
            server.streams().written([&destination].into_iter());
            Ok(())
        }
        Err(error) => {
            server.collectives().mark_broken(group, error.clone());
            server
                .streams()
                .taint(error.clone(), [&destination].into_iter());
            Err(error)
        }
    }
}

/// Runs `collective` on the collective stream over `bindings`, once the compute stream has
/// written them, keeping them borrowed until the compute stream waits as `wait` says.
///
/// # Errors
///
/// A group never joined, a binding that names no live allocation, a wait the driver refuses,
/// and whatever `collective` returns.
fn run<S: CollectiveServer, const N: usize>(
    server: &mut S,
    stream_id: StreamId,
    group: &CommunicationId,
    bindings: [BufferBinding; N],
    wait: ComputeWait,
    timing: Option<CommunicationTiming>,
    collective: impl FnOnce(
        &<S::Driver as CollectiveDriver>::Communicator,
        [&DeviceResource<S::Driver>; N],
        StreamSignal<S::Driver>,
    ) -> Result<(), ServerError>,
) -> Result<(), ServerError> {
    let comm = *server.collectives().get(group)?;
    let comm_stream = server.comm_stream();
    let marks = CommunicationMarks::new(server, timing);

    let mut command = server.command(stream_id, bindings.iter())?;
    let mut resources = Vec::with_capacity(N);
    for binding in bindings {
        resources.push(command.resource(binding)?);
    }
    let mut marks = marks.unless_capturing(command.stream());
    let stream = command.stream().signal();

    marks.hand_over(comm_stream, stream)?;
    collective(&comm, core::array::from_fn(|i| &resources[i]), comm_stream)?;
    marks.ended(comm_stream, stream, wait)?;
    // Dropping the command releases bindings borrowed from other streams at this point of
    // the compute stream, which only waits for the collective under `ComputeWait::Now`.
    drop(command);
    Ok(())
}

/// The send itself, so every way it can fail reaches the peer through one path in
/// [`CollectiveServer::send`].
///
/// # Errors
///
/// A source some failure left unwritten, and the driver's refusal to join or send.
fn post_send<S: CollectiveServer>(
    server: &mut S,
    desc: CopyDescriptor,
    stream_id: StreamId,
    device_id_dst: DeviceId,
    report: &SendReport,
    timing: Option<CommunicationTiming>,
) -> Result<(), ServerError> {
    // Joined before anything that can fail: on a pair's first transfer the
    // peer's thread waits in its own join until this one arrives.
    let comm = *server.collectives().transfers_with(device_id_dst)?;
    let devices = pair(server.device_id(), device_id_dst);
    let peer = server.collectives().peer_rank(&devices)?;
    let transfers = server.transfer_stream();
    let marks = CommunicationMarks::new(server, timing);

    // The send reads the source, so it is worth no more than the work that wrote it. Skipping
    // this hands the peer stale bytes on a handle that carries no claim over there: the
    // failure would be laundered across the device boundary.
    server
        .streams()
        .ensure_written([&desc.handle].into_iter())?;

    let binding = desc.handle.clone();
    let size = binding.size_in_used();

    // A command on the source server retrieves the resource from the right
    // memory pool, and aligns the current stream with the one the binding
    // was allocated on.
    let mut command = server.command(stream_id, [&desc.handle].into_iter())?;
    let resource = command.resource(binding)?;
    let mut marks = marks.unless_capturing(command.stream());
    let stream = command.stream().signal();

    marks.hand_over(transfers, stream)?;
    let bytes = S::Driver::elements(BYTES, size)?;
    report.will_post();
    S::Driver::send(&comm, &resource, bytes, peer, transfers)?;
    // This stream's pool may reuse the source from here, and dropping the command releases one
    // borrowed from another stream once this stream gets here: both wait for the send.
    marks.ended(transfers, stream, ComputeWait::Now)?;
    drop(command);
    Ok(())
}

/// The receive itself, so every way it can fail settles the destination through one path in
/// [`CollectiveServer::recv`].
///
/// # Errors
///
/// A reservation the device cannot back, a group never joined, a peer outside it, and the
/// driver's refusal to enqueue.
fn receive<S: CollectiveServer>(
    server: &mut S,
    handle: Handle,
    stream_id: StreamId,
    device_id_src: DeviceId,
    outcome: SendOutcome,
    timing: Option<CommunicationTiming>,
) -> Result<(), ServerError> {
    // Joined before anything that can fail or relocate: on a pair's first
    // transfer the peer's thread waits in its own join until this one arrives.
    let comm = *server.collectives().transfers_with(device_id_src)?;
    let devices = pair(server.device_id(), device_id_src);
    let peer = server.collectives().peer_rank(&devices)?;
    let size = handle.size_in_used();
    let marks = CommunicationMarks::new(server, timing);

    // A command on the destination server reserves the memory the incoming
    // data lands in.
    let mut command = server.command(stream_id, [].into_iter())?;
    let memory = command.reserve(handle.size())?;
    command.bind(memory, handle.memory.clone())?;
    let resource = command.resource(handle.binding())?;
    let mut marks = marks.unless_capturing(command.stream());
    let stream = command.stream().signal();
    drop(command);

    // After the reservation, so a refusal has memory to taint. A send that refused posts nothing,
    // and a recv posted without it would wait forever.
    outcome.wait()?;

    // Compute work queued before the reservation may still use its memory, and compute work
    // queued after it reads what arrives.
    let transfers = server.transfer_stream();
    marks.hand_over(transfers, stream)?;
    let bytes = S::Driver::elements(BYTES, size)?;
    S::Driver::recv(&comm, &resource, bytes, peer, transfers)?;
    marks.ended(transfers, stream, ComputeWait::Now)
}

/// The marks that time one communication part on its own stream and a compute stream's wait on
/// it, as a [`CommunicationTiming`] asks for them.
///
/// Timed, the events that order the two streams are these marks rather than fresh ones, so a
/// part's compute stream carries no event it would not carry untimed: on HIP, one more timing
/// event recorded there before its wait costs a decode step a quarter of its time. Taken before
/// the part is queued, so that recording them borrows nothing from the server. Timing never fails
/// a communication: a mark the device will not record leaves it ordered as an untimed one, and
/// its windows read back empty.
struct CommunicationMarks<S: CollectiveServer> {
    timing: CommunicationTiming,
    /// Where the compute stream reaches its wait. For a part, where it hands the part its inputs:
    /// nothing is queued on it in between.
    reached: Option<Mark<S::Events>>,
    start: Option<Mark<S::Events>>,
    end: Option<Mark<S::Events>>,
}

impl<S: CollectiveServer> CommunicationMarks<S> {
    /// The marks `timing` asks for, reaching for `server`'s profiler only when it asks for any.
    fn new(server: &mut S, timing: Option<CommunicationTiming>) -> Self {
        let Some(timing) = timing else {
            return Self::untimed();
        };
        Self::taken(server.profiler(), timing).unwrap_or_else(Self::failed)
    }

    fn taken(
        profiler: &mut EventProfiler<S::Events>,
        timing: CommunicationTiming,
    ) -> Result<Self, DriverError> {
        let mut mark = |wanted: bool| wanted.then(|| profiler.mark()).transpose();
        Ok(Self {
            reached: mark(true)?,
            start: mark(timing.times_part())?,
            end: mark(true)?,
            timing,
        })
    }

    fn untimed() -> Self {
        Self {
            timing: CommunicationTiming::default(),
            reached: None,
            start: None,
            end: None,
        }
    }

    fn failed(error: DriverError) -> Self {
        log::warn!(
            "A communication runs untimed: {} refused a timing event ({error})",
            <S::Events as EventApi>::BACKEND
        );
        Self::untimed()
    }

    /// These marks, unless `stream` is capturing a graph, whose recording takes no timing event.
    fn unless_capturing(self, stream: &mut impl DeviceStream) -> Self {
        match stream.capturing().is_recording() {
            true => Self::untimed(),
            false => self,
        }
    }

    /// Makes `part_stream` wait for the work queued on `compute` so far, then starts the part
    /// there.
    ///
    /// # Errors
    ///
    /// The driver's refusal to order the two.
    fn hand_over(
        &mut self,
        part_stream: StreamSignal<S::Driver>,
        compute: StreamSignal<S::Driver>,
    ) -> Result<(), ServerError> {
        if let Some(reached) = &self.reached {
            match reached
                .record(compute)
                .and_then(|()| reached.wait_async(part_stream))
            {
                Ok(()) => {
                    self.started(part_stream);
                    return Ok(());
                }
                Err(error) => *self = Self::failed(error),
            }
        }
        S::Driver::wait_for(part_stream, compute)
    }

    /// The compute stream `compute` reaches its wait for the collectives queued so far.
    fn reach(&mut self, compute: StreamSignal<S::Driver>) {
        if let Some(reached) = &self.reached
            && let Err(error) = reached.record(compute)
        {
            *self = Self::failed(error);
        }
    }

    fn started(&mut self, part_stream: StreamSignal<S::Driver>) {
        if let Some(start) = &self.start
            && let Err(error) = start.record(part_stream)
        {
            *self = Self::failed(error);
        }
    }

    /// The part ends at the current position of `part_stream`, and under `ComputeWait::Now`
    /// `compute` waits for it there, which sends the windows.
    ///
    /// # Errors
    ///
    /// The driver's refusal to order the two.
    fn ended(
        &mut self,
        part_stream: StreamSignal<S::Driver>,
        compute: StreamSignal<S::Driver>,
        wait: ComputeWait,
    ) -> Result<(), ServerError> {
        if let Some(end) = &self.end {
            let ordered = end.record(part_stream).and_then(|()| match wait {
                ComputeWait::Now => end.wait_async(compute),
                ComputeWait::AtSync => Ok(()),
            });
            match ordered {
                Ok(()) => {
                    self.measured(wait);
                    return Ok(());
                }
                Err(error) => *self = Self::failed(error),
            }
        }
        match wait {
            ComputeWait::Now => S::Driver::wait_for(compute, part_stream),
            ComputeWait::AtSync => Ok(()),
        }
    }

    fn measured(&mut self, wait: ComputeWait) {
        let Some(end) = &self.end else {
            return;
        };
        if let Some(start) = &self.start {
            self.timing.part_measured(EventProfiler::span(start, end));
        }
        if let (ComputeWait::Now, Some(reached)) = (wait, &self.reached) {
            self.timing.wait_measured(EventProfiler::span(reached, end));
        }
    }
}

/// The two devices of a peer-to-peer transfer, sorted.
///
/// Sorted because a rank is a position: `send` and `recv` are the two sides of
/// one transfer and have to agree on which device is which.
fn pair(this: DeviceId, peer: DeviceId) -> Vec<DeviceId> {
    let mut devices = vec![this, peer];
    devices.sort();
    devices
}
