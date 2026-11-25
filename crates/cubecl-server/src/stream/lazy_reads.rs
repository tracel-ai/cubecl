//! The lazy reads a device keeps, which work about to write their memory starts first.

use crate::memory_management::ManagedMemoryId;
use crate::server::{BufferBinding, CopyDescriptor, LazyRead, LazyReadCount, ServerError};
use crate::stream::{ExecuteScope, FailureStore, StreamCapture, StreamFence, WriteScoped};
use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;

type Copy = DynFut<Result<Vec<Bytes>, ServerError>>;

/// The lazy reads a device keeps, until work about to write their memory starts them and their
/// copies are done, or their bytes let them go.
#[derive(Debug, Default)]
pub(crate) struct LazyReads {
    /// The reads kept, started and accessed, which spares every write the lookup while none are.
    count: LazyReadCount,
    /// Keyed usage: the reads of each allocation, looked up by the allocation a write names.
    kept: HashMap<ManagedMemoryId, KeptReads>,
    /// How many allocations `kept` named when it last let go of the reads nothing holds.
    swept_at: usize,
    /// Keyed usage, by read through a scan of the few held at once: the copies of started reads,
    /// until an access takes one or its read drops.
    copies: Vec<StartedCopy>,
    /// Arrival usage: the copies whose read dropped, until the fence past each is reached, since
    /// letting go of a copy in flight waits for it.
    abandoned: Vec<StartedCopy>,
    /// What [`LazyReadCount::let_go_so_far`] said when `copies` was last scanned for dropped reads.
    let_go_at_scan: usize,
}

/// The reads of one allocation.
#[derive(Debug, Default)]
struct KeptReads {
    /// Arrival usage: scanned by every write to the allocation, and pruned of the reads no write
    /// would find once it reaches `prune_at`.
    pub reads: Vec<KeptRead>,
    /// Twice the reads left by the last pruning, so pruning costs linear time over the pushes.
    pub prune_at: usize,
}

#[derive(Debug)]
struct KeptRead {
    pub read: Weak<LazyRead>,
    /// The stream the read was made on until it is started, then the stream its copy runs on.
    pub stream: StreamId,
    /// Past the work on `stream` when the read was made, while it is kept and its server gives
    /// one, so the copy can run behind exactly that work on the writer's stream.
    pub made_at: Option<Arc<dyn StreamFence>>,
    /// Past the copy once the read is started, when its server gives one.
    pub fence: Option<Arc<dyn StreamFence>>,
    /// The memory an access copies on its own while the read stays kept, which work writing
    /// it waits on until the copy is done or the access has its bytes.
    pub accessed: Option<(Range<u64>, Weak<AccessCopy>)>,
}

/// The copy of a started read, kept for its first access.
struct StartedCopy {
    pub read: Weak<LazyRead>,
    /// Past the copy, when its server gave one.
    pub fence: Option<Arc<dyn StreamFence>>,
    pub copy: Copy,
}

/// An access's copy of a kept read's bytes, counted on its memory so work writing the memory
/// finds it, until the access has its bytes.
#[derive(Debug)]
struct AccessCopy {
    read: Arc<LazyRead>,
    count: LazyReadCount,
}

/// A lazy read's copy that may still be running.
#[derive(Debug)]
struct CopyInFlight {
    /// The stream it runs on.
    pub stream: StreamId,
    /// Past the copy, when its server gave one; otherwise only the stream orders it.
    pub fence: Option<Arc<dyn StreamFence>>,
}

/// A kept read work is about to overwrite, which it starts first.
#[derive(Debug)]
struct ReadToStart {
    pub read: Arc<LazyRead>,
    /// The stream it was made on.
    pub made_on: StreamId,
    /// Past the work on `made_on` when it was made, when its server gave one.
    pub made_at: Option<Arc<dyn StreamFence>>,
}

/// What work about to write memory lazy reads hold has to do first.
#[derive(Debug, Default)]
struct Overwritten {
    /// Arrival usage: the kept reads to start.
    pub reads: Vec<ReadToStart>,
    /// Arrival usage: the copies of the memory that may still be running.
    pub in_flight: Vec<CopyInFlight>,
}

impl LazyReads {
    /// Whether no read is kept, started or accessed and no copy is held, so a write has
    /// nothing to start or wait for.
    fn is_idle(&self) -> bool {
        self.count.is_zero() && self.copies.is_empty() && self.abandoned.is_empty()
    }

    /// Keep `read`, made on `stream` where `made_at` fences, until work about to write its
    /// memory takes it.
    fn keep(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        made_at: Option<Arc<dyn StreamFence>>,
    ) {
        self.hold(
            read,
            KeptRead {
                read: Arc::downgrade(read),
                stream,
                made_at,
                fence: None,
                accessed: None,
            },
        );
    }

    /// Hold `copy`, of `accessed`, bytes of `read` an access reads on `stream` while it stays
    /// kept, so work writing them waits for it, on `fence` when there is one.
    fn accessed(
        &mut self,
        read: &Arc<LazyRead>,
        accessed: Range<u64>,
        copy: &Arc<AccessCopy>,
        stream: StreamId,
        fence: Option<Arc<dyn StreamFence>>,
    ) {
        self.hold(
            read,
            KeptRead {
                read: Arc::downgrade(read),
                stream,
                made_at: None,
                fence,
                accessed: Some((accessed, Arc::downgrade(copy))),
            },
        );
    }

    /// Hold `read`, started on `stream`, until its copy is done, so work writing its memory
    /// from another stream waits for it, on `fence` when there is one, and keep `copy` for its
    /// first access.
    fn started(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        fence: Option<Arc<dyn StreamFence>>,
        copy: Copy,
    ) {
        self.copies.push(StartedCopy {
            read: Arc::downgrade(read),
            fence: fence.clone(),
            copy,
        });
        self.hold(
            read,
            KeptRead {
                read: Arc::downgrade(read),
                stream,
                made_at: None,
                fence,
                accessed: None,
            },
        );
    }

    /// The copy of `read` this keeps, which it then lets go of.
    fn take_copy(&mut self, read: &Arc<LazyRead>) -> Option<Copy> {
        let index = self.copies.iter().position(|copy| copy.is_of(read))?;
        Some(self.copies.swap_remove(index).copy)
    }

    /// Let go of the copies no access will take, once they are done, scanning the copies of
    /// live reads only when a read was let go of since the last scan.
    fn drop_done_copies(&mut self) {
        let let_go = self.count.let_go_so_far();
        if let_go != self.let_go_at_scan {
            self.let_go_at_scan = let_go;
            let (abandoned, copies) = core::mem::take(&mut self.copies)
                .into_iter()
                .partition(StartedCopy::is_abandoned);
            self.copies = copies;
            self.abandoned.extend(abandoned);
        }
        self.abandoned.retain(StartedCopy::is_in_flight);
    }

    /// Where the kept `read` was made, when it was made on a stream other than `stream`: that
    /// stream, and the fence past the work on it then when its server gave one.
    fn made_elsewhere(
        &self,
        read: &Arc<LazyRead>,
        stream: StreamId,
    ) -> Option<(StreamId, Option<Arc<dyn StreamFence>>)> {
        self.kept
            .get(&read.to_source().handle.memory.id())?
            .reads
            .iter()
            .find(|kept| {
                kept.accessed.is_none() && core::ptr::eq(kept.read.as_ptr(), Arc::as_ptr(read))
            })
            .filter(|kept| kept.stream != stream)
            .map(|kept| (kept.stream, kept.made_at.clone()))
    }

    /// What work about to write `written` has to do first: the kept reads whose bytes it
    /// overlaps, which are no longer kept, and the streams copies of them are in flight on.
    fn take_overwritten(&mut self, written: &[BufferBinding]) -> Overwritten {
        let mut overwritten = Overwritten::default();
        let written = written
            .iter()
            .filter(|binding| binding.memory.descriptor().has_lazy_reads());
        for binding in written {
            let id = binding.memory.id();
            let Some(reads) = self.kept.get_mut(&id) else {
                continue;
            };
            reads
                .reads
                .retain(|kept| kept.is_left_by(binding, &mut overwritten));
            if reads.reads.is_empty() {
                self.kept.remove(&id);
            }
        }
        overwritten
    }

    fn hold(&mut self, read: &Arc<LazyRead>, kept: KeptRead) {
        let reads = self
            .kept
            .entry(read.to_source().handle.memory.id())
            .or_default();
        reads.prune();
        reads.reads.push(kept);
        self.sweep_when_grown();
    }

    /// Let go of the reads nothing holds, keeps or copies anymore, once the allocations named
    /// have doubled since the last time.
    fn sweep_when_grown(&mut self) {
        if self.kept.len() > 2 * self.swept_at.max(16) {
            self.sweep();
        }
    }

    fn sweep(&mut self) {
        self.kept.retain(|_, reads| {
            reads.reads.retain(KeptRead::is_live);
            !reads.reads.is_empty()
        });
        self.swept_at = self.kept.len();
    }
}

impl KeptReads {
    /// Let go of the reads no write would find, once there are `prune_at` of them.
    fn prune(&mut self) {
        if self.reads.len() >= self.prune_at.max(8) {
            self.reads.retain(KeptRead::is_live);
            self.prune_at = 2 * self.reads.len();
        }
    }
}

impl KeptRead {
    /// Whether a write may still have to start its read or wait for its copy.
    fn is_live(&self) -> bool {
        if self.fence.as_ref().is_some_and(|fence| fence.is_reached()) {
            return false;
        }
        match &self.accessed {
            Some((_, copy)) => copy.strong_count() > 0,
            None => self
                .read
                .upgrade()
                .is_some_and(|read| read.is_kept() || read.is_in_flight()),
        }
    }

    /// Whether it stays after `written` is written, adding to `overwritten` the read to start
    /// or the copy to wait for that the write overlaps.
    fn is_left_by(&self, written: &BufferBinding, overwritten: &mut Overwritten) -> bool {
        if !self.is_live() {
            return false;
        }
        if let Some((accessed, _)) = &self.accessed {
            if written.overlaps(accessed) {
                overwritten.in_flight.push(self.in_flight());
            }
            return true;
        }
        let Some(read) = self.read.upgrade() else {
            return false;
        };
        if !read.is_overwritten_by(written) {
            return true;
        }
        let is_kept = read.is_kept();
        match is_kept {
            true => overwritten.reads.push(ReadToStart {
                read,
                made_on: self.stream,
                made_at: self.made_at.clone(),
            }),
            false => overwritten.in_flight.push(self.in_flight()),
        }
        !is_kept
    }

    fn in_flight(&self) -> CopyInFlight {
        CopyInFlight {
            stream: self.stream,
            fence: self.fence.clone(),
        }
    }
}

impl StartedCopy {
    fn is_of(&self, read: &Arc<LazyRead>) -> bool {
        core::ptr::eq(self.read.as_ptr(), Arc::as_ptr(read)) && self.read.strong_count() > 0
    }

    /// Whether its read dropped, so no access will take it.
    fn is_abandoned(&self) -> bool {
        self.read.strong_count() == 0
    }

    /// Whether letting go of it would wait for it.
    fn is_in_flight(&self) -> bool {
        self.fence.as_ref().is_some_and(|fence| !fence.is_reached())
    }
}

impl core::fmt::Debug for StartedCopy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StartedCopy")
            .field("fence", &self.fence)
            .finish()
    }
}

impl AccessCopy {
    fn new(read: &Arc<LazyRead>, count: &LazyReadCount) -> Arc<Self> {
        count.add(read.descriptor());
        Arc::new(Self {
            read: read.clone(),
            count: count.clone(),
        })
    }
}

impl Drop for AccessCopy {
    fn drop(&mut self) {
        self.count.remove(self.read.descriptor());
    }
}

fn lazy_reads<S: WriteScoped>(server: &mut S) -> &mut LazyReads {
    server.write_streams().split().1.lazy_reads()
}

/// See [`WriteScoped::keep_lazy_read`].
pub(crate) fn keep<S: WriteScoped>(
    server: &mut S,
    read: Arc<LazyRead>,
    stream: StreamId,
) -> Result<(), ServerError> {
    if is_recording(server.capturing(stream)) {
        return Err(ServerError::graph_state(
            "a lazy read is refused on a stream recording a graph: its copy would be recorded \
             into the graph rather than run",
        ));
    }
    let source = read.to_source();
    server
        .write_streams()
        .ensure_written(core::iter::once(&source.handle))?;
    let reads = lazy_reads(server);
    reads.drop_done_copies();
    read.keep(&reads.count);
    if source.handle.memory.descriptor().is_shared_with_peers() {
        start(server, &read, stream);
        return Ok(());
    }
    let made_at = server.fence_stream(stream);
    lazy_reads(server).keep(&read, stream, made_at);
    Ok(())
}

/// See [`WriteScoped::read_lazy_access`].
pub(crate) fn access<S: WriteScoped>(
    server: &mut S,
    read: Arc<LazyRead>,
    source: CopyDescriptor,
    stream: StreamId,
) -> Copy {
    let accessed = source.handle.range();
    if !is_recording(server.capturing(stream)) {
        order_after_made(server, &read, stream);
    }
    let Some(copy) = server.read_lazy_copy(vec![source], stream) else {
        return refused("this server keeps no lazy read");
    };
    if !read.is_kept() || is_recording(server.capturing(stream)) {
        return copy;
    }
    let fence = server.fence_stream(stream);
    let reads = lazy_reads(server);
    let in_flight = AccessCopy::new(&read, &reads.count);
    reads.accessed(&read, accessed, &in_flight, stream, fence);
    Box::pin(async move {
        let copied = copy.await;
        core::mem::drop(in_flight);
        copied
    })
}

/// See [`WriteScoped::take_started_copy`].
pub(crate) fn take_copy<S: WriteScoped>(server: &mut S, read: &Arc<LazyRead>) -> Copy {
    lazy_reads(server)
        .take_copy(read)
        .unwrap_or_else(|| refused("this server holds no copy of the lazy read"))
}

/// Start every lazy read `server` keeps of memory `written` names, and order work on `writer`
/// after the copies of it, ahead of that work writing it.
///
/// Does nothing while `writer` records a graph: the recorded work writes nothing until a
/// replay, whose own write scope starts the reads then.
pub(crate) fn start_overwritten_reads<S: WriteScoped>(
    server: &mut S,
    writer: StreamId,
    written: &[BufferBinding],
) {
    let reads = lazy_reads(server);
    if written.is_empty() || reads.is_idle() {
        return;
    }
    reads.drop_done_copies();
    let overlaps_a_read = written
        .iter()
        .any(|binding| binding.memory.descriptor().has_lazy_reads());
    if !overlaps_a_read || is_recording(server.capturing(writer)) {
        return;
    }
    let overwritten = lazy_reads(server).take_overwritten(written);
    let mut in_flight = overwritten.in_flight;
    for kept in overwritten.reads {
        let stream = copy_stream(server, &kept, writer);
        in_flight.extend(start(server, &kept.read, stream));
    }
    for copy in in_flight.iter().filter(|copy| copy.stream != writer) {
        wait_for_copy(server, writer, copy);
    }
}

/// See [`WriteScoped::start_reads_before_sharing`].
pub(crate) fn start_before_sharing<S: WriteScoped>(
    server: &mut S,
    binding: &BufferBinding,
) -> Result<(), ServerError> {
    let descriptor = binding.memory.descriptor();
    if descriptor.has_lazy_reads() && is_recording(server.capturing(binding.stream)) {
        return Err(ServerError::graph_state(
            "memory a lazy read holds is refused to peers while its stream records a graph: \
             the read's copy would be recorded into the graph rather than run",
        ));
    }
    descriptor.share_with_peers();
    start_overwritten_reads(server, binding.stream, core::slice::from_ref(binding));
    Ok(())
}

/// The stream `kept`'s copy runs on: the writer, behind the reader's work up to the read and
/// none after it, which may itself wait on the writer.
fn copy_stream<S: WriteScoped>(server: &mut S, kept: &ReadToStart, writer: StreamId) -> StreamId {
    match (&kept.made_at, kept.made_on == writer) {
        (_, true) => writer,
        (Some(made_at), false) => behind(server, writer, made_at.as_ref()),
        (None, false) => reader_stream(server, kept.made_on, writer),
    }
}

/// Order work on `stream` after the work on the stream the kept `read` was made on, up to it.
fn order_after_made<S: WriteScoped>(server: &mut S, read: &Arc<LazyRead>, stream: StreamId) {
    if !read.is_kept() {
        return;
    }
    match lazy_reads(server).made_elsewhere(read, stream) {
        Some((_, Some(made_at))) => wait_past(server, stream, made_at.as_ref()),
        Some((made_on, None)) => server.wait_before_write(stream, made_on),
        None => {}
    }
}

/// `writer`, ordered after `made_at`.
fn behind<S: WriteScoped>(server: &mut S, writer: StreamId, made_at: &dyn StreamFence) -> StreamId {
    wait_past(server, writer, made_at);
    writer
}

/// Order work on `stream` after `fence`, unless it is already reached.
fn wait_past<S: WriteScoped>(server: &mut S, stream: StreamId, fence: &dyn StreamFence) {
    if !fence.is_reached() {
        server.wait_for_fence(stream, fence);
    }
}

/// The reader's stream, unless it records a graph, where a copy would run at every replay
/// instead.
fn reader_stream<S: WriteScoped>(server: &mut S, reader: StreamId, writer: StreamId) -> StreamId {
    match is_recording(server.capturing(reader)) {
        true => writer,
        false => reader,
    }
}

fn wait_for_copy<S: WriteScoped>(server: &mut S, writer: StreamId, copy: &CopyInFlight) {
    match &copy.fence {
        Some(fence) => server.wait_for_fence(writer, fence.as_ref()),
        None => server.wait_before_write(writer, copy.stream),
    }
}

/// Start `read` on `stream`, in a scope of its own, holding its copy with a fence past it when
/// `server` gives one.
fn start<S: WriteScoped>(
    server: &mut S,
    read: &Arc<LazyRead>,
    stream: StreamId,
) -> Option<CopyInFlight> {
    let (sources, offsets) = read.to_held_sources();
    if sources.is_empty() {
        return None;
    }
    // A scope that writes nothing cannot fail, and its body reports a failed copy through it.
    let copy = ExecuteScope::over(server, stream, Vec::new())
        .execute(|server| Ok::<_, ServerError>(server.read_lazy_copy(sources, stream)))
        .into_result()
        .ok()
        .flatten()?;
    let fence = server.fence_stream(stream);
    read.start(offsets);
    lazy_reads(server).started(read, stream, fence.clone(), copy);
    Some(CopyInFlight { stream, fence })
}

fn refused(reason: &'static str) -> Copy {
    Box::pin(async move {
        Err(ServerError::Generic {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        })
    })
}

fn is_recording(capture: Option<&mut StreamCapture>) -> bool {
    capture.is_some_and(|capture| capture.is_recording())
}
