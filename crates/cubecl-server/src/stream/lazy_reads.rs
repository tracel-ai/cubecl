//! The lazy reads a device keeps, which work about to write their memory starts first.

use crate::device_events::{Event, EventApi};
use crate::memory_management::ManagedMemoryId;
use crate::server::{BufferBinding, LazyRead, RetireCopy, ServerError};
use crate::stream::{FailureStore, StreamCapture, WriteScoped};
use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::collections::HashMap;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::Mutex;

/// The lazy reads a device keeps, until work about to write their memory starts them and their
/// copies resolve, or their bytes let them go.
#[derive(Debug, Default)]
pub struct LazyReads {
    /// Keyed usage: the reads of each allocation, looked up by the allocation a write names.
    kept: HashMap<ManagedMemoryId, Vec<KeptRead>>,
    /// How many allocations `kept` named when it last let go of the reads nothing holds.
    swept_at: usize,
    /// The copies of started reads dropped unread, until their fences are reached and the next
    /// lazy read is kept or overwritten: letting go of a copy in flight waits for it, which
    /// would block whoever dropped the read.
    retired: Retired,
}

type RetiredCopy = (
    Arc<dyn StreamFence>,
    DynFut<Result<Vec<Bytes>, ServerError>>,
);

#[derive(Clone, Default)]
struct Retired(Arc<Mutex<Vec<RetiredCopy>>>);

impl core::fmt::Debug for Retired {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} retired copies", self.0.lock().len())
    }
}

#[derive(Debug)]
struct KeptRead {
    read: Weak<LazyRead>,
    /// The stream the read was made on until it is started, then the stream its copy runs on.
    stream: StreamId,
    /// Past the work on `stream` when the read was made, while it is kept and its server gives
    /// one, so the copy can run behind exactly that work on the writer's stream.
    made_at: Option<Arc<dyn StreamFence>>,
    /// Past the copy once the read is started, when its server gives one.
    fence: Option<Arc<dyn StreamFence>>,
    /// The memory an access copies on its own while the read stays kept, which work writing
    /// it waits on until the copy is done or the access has its bytes.
    accessed: Option<(Range<u64>, Weak<AccessCopy>)>,
}

/// An access's copy of a kept read's bytes, counted on its memory so work writing the memory
/// finds it, until the access has its bytes.
#[derive(Debug)]
struct AccessCopy(Arc<LazyRead>);

impl AccessCopy {
    fn new(read: &Arc<LazyRead>) -> Arc<Self> {
        read.descriptor().lazy_read_kept();
        Arc::new(Self(read.clone()))
    }
}

impl Drop for AccessCopy {
    fn drop(&mut self) {
        self.0.descriptor().lazy_read_let_go();
    }
}

/// A point on a stream, where a lazy read was made or just past its copy, which work on another
/// stream waits on rather than on all the work queued there since.
pub trait StreamFence: core::any::Any + Send + Sync + core::fmt::Debug {
    /// Whether the device has finished the work before it.
    fn is_reached(&self) -> bool;
}

impl<A: EventApi> StreamFence for Event<A>
where
    A::Event: Send + Sync,
{
    fn is_reached(&self) -> bool {
        Event::is_reached(self)
    }
}

/// A lazy read's copy that may still be running.
#[derive(Debug)]
pub struct CopyInFlight {
    /// The stream it runs on.
    pub stream: StreamId,
    /// Past the copy, when its server gave one; otherwise only the stream orders it.
    pub fence: Option<Arc<dyn StreamFence>>,
}

/// A kept read work is about to overwrite, which it starts first.
#[derive(Debug)]
pub struct ReadToStart {
    /// The read.
    pub read: Arc<LazyRead>,
    /// The stream it was made on.
    pub made_on: StreamId,
    /// Past the work on `made_on` when it was made, when its server gave one.
    pub made_at: Option<Arc<dyn StreamFence>>,
}

/// What work about to write memory lazy reads hold has to do first.
#[derive(Debug, Default)]
pub struct Overwritten {
    /// The kept reads to start.
    pub reads: Vec<ReadToStart>,
    /// The started copies of the memory that may still be running.
    pub in_flight: Vec<CopyInFlight>,
}

impl LazyReads {
    /// Keep `read`, made on `stream` where `made_at` fences, until work about to write its
    /// memory takes it.
    pub fn keep(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        made_at: Option<Arc<dyn StreamFence>>,
    ) {
        read.keep();
        self.hold(read, stream, made_at, None, None);
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
            stream,
            None,
            fence,
            Some((accessed, Arc::downgrade(copy))),
        );
    }

    /// Hold `read`, started on `stream`, until its copy is done, so work writing its memory
    /// from another stream waits for it, on `fence` when there is one.
    pub fn started(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        fence: Option<Arc<dyn StreamFence>>,
    ) {
        let entry = self
            .kept
            .get_mut(&read.to_source().handle.memory.id())
            .and_then(|reads| {
                reads.iter_mut().find(|kept| {
                    kept.accessed.is_none() && core::ptr::eq(kept.read.as_ptr(), Arc::as_ptr(read))
                })
            });
        match entry {
            Some(kept) => {
                kept.stream = stream;
                kept.made_at = None;
                kept.fence = fence;
            }
            None => self.hold(read, stream, None, fence, None),
        }
    }

    /// What work about to write `written` has to do first: the kept reads whose bytes it
    /// overlaps, which are no longer kept, and the streams copies of them are in flight on.
    pub fn take_overwritten(&mut self, written: &[BufferBinding]) -> Overwritten {
        self.drop_retired();
        let mut overwritten = Overwritten::default();
        let written = written
            .iter()
            .filter(|binding| binding.memory.descriptor().has_lazy_reads());
        for binding in written {
            let id = binding.memory.id();
            let Some(reads) = self.kept.get_mut(&id) else {
                continue;
            };
            reads.retain(|kept| {
                if !is_live(kept) {
                    return false;
                }
                if let Some((accessed, _)) = &kept.accessed {
                    let written = binding.range();
                    if accessed.start < written.end && written.start < accessed.end {
                        overwritten.in_flight.push(CopyInFlight {
                            stream: kept.stream,
                            fence: kept.fence.clone(),
                        });
                    }
                    return true;
                }
                let Some(read) = kept.read.upgrade() else {
                    return false;
                };
                if !read.is_overwritten_by(binding) {
                    return true;
                }
                let is_kept = read.is_kept();
                match is_kept {
                    true => overwritten.reads.push(ReadToStart {
                        read,
                        made_on: kept.stream,
                        made_at: kept.made_at.clone(),
                    }),
                    false => overwritten.in_flight.push(CopyInFlight {
                        stream: kept.stream,
                        fence: kept.fence.clone(),
                    }),
                }
                !is_kept
            });
            if reads.is_empty() {
                self.kept.remove(&id);
            }
        }
        overwritten
    }

    fn hold(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        made_at: Option<Arc<dyn StreamFence>>,
        fence: Option<Arc<dyn StreamFence>>,
        accessed: Option<(Range<u64>, Weak<AccessCopy>)>,
    ) {
        self.drop_retired();
        let reads = self
            .kept
            .entry(read.to_source().handle.memory.id())
            .or_default();
        // Pruned at each doubling, so repeated reads of one buffer stay bounded at linear cost.
        if reads.len() >= 8 && reads.len().is_power_of_two() {
            reads.retain(is_live);
        }
        reads.push(KeptRead {
            read: Arc::downgrade(read),
            stream,
            made_at,
            fence,
            accessed,
        });
        if self.kept.len() > 2 * self.swept_at.max(16) {
            self.sweep();
        }
    }

    /// What a started read hands its copy to if it drops unread: the copy is let go of here
    /// once `fence`, past it, is reached.
    fn retire(&self, fence: Option<Arc<dyn StreamFence>>) -> RetireCopy {
        match fence {
            Some(fence) => {
                let retired = self.retired.clone();
                Box::new(move |copy| match fence.is_reached() {
                    true => drop(copy),
                    false => retired.0.lock().push((fence, copy)),
                })
            }
            None => Box::new(core::mem::drop),
        }
    }

    fn drop_retired(&mut self) {
        let done: Vec<RetiredCopy> = {
            let mut retired = self.retired.0.lock();
            if retired.is_empty() {
                return;
            }
            let (done, pending) = core::mem::take(&mut *retired)
                .into_iter()
                .partition(|(fence, _)| fence.is_reached());
            *retired = pending;
            done
        };
        core::mem::drop(done);
    }

    /// Let go of the reads nothing holds, keeps or copies anymore, which no write would find.
    fn sweep(&mut self) {
        self.kept.retain(|_, reads| {
            reads.retain(is_live);
            !reads.is_empty()
        });
        self.swept_at = self.kept.len();
    }
}

/// Whether a write may still have to start `kept`'s read or wait for its copy.
fn is_live(kept: &KeptRead) -> bool {
    if kept.fence.as_ref().is_some_and(|fence| fence.is_reached()) {
        return false;
    }
    match &kept.accessed {
        Some((_, copy)) => copy.strong_count() > 0,
        None => kept
            .read
            .upgrade()
            .is_some_and(|read| read.is_kept() || read.is_in_flight()),
    }
}

/// Keep `read`, made on `stream`, for work `server` queues to write its memory to start first.
///
/// # Errors
///
/// A refusal on a stream recording a graph, where the read's copy would be recorded rather
/// than run, and the failure its memory carries.
pub fn keep_lazy_read<S: WriteScoped>(
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
    let made_at = server.fence_stream(stream);
    server
        .write_streams()
        .lazy_reads()
        .keep(&read, stream, made_at);
    if source.handle.memory.descriptor().is_shared_with_peers() {
        start(server, &read, stream);
    }
    Ok(())
}

/// `copy`, an access's read of `accessed`, bytes of `read`, just queued on `stream`, held while
/// `server` keeps the read so work writing those bytes afterwards waits for it as for a copy a
/// write started, until it is done or resolves.
///
/// Holds nothing while `stream` records a graph, where the copy is recorded rather than run.
pub fn lazy_read_accessed<S: WriteScoped>(
    server: &mut S,
    read: &Arc<LazyRead>,
    accessed: Range<u64>,
    stream: StreamId,
    copy: DynFut<Result<Vec<Bytes>, ServerError>>,
) -> DynFut<Result<Vec<Bytes>, ServerError>> {
    if !read.is_kept() || is_recording(server.capturing(stream)) {
        return copy;
    }
    let fence = server.fence_stream(stream);
    let in_flight = AccessCopy::new(read);
    server
        .write_streams()
        .lazy_reads()
        .accessed(read, accessed, &in_flight, stream, fence);
    Box::pin(async move {
        let copied = copy.await;
        core::mem::drop(in_flight);
        copied
    })
}

/// Start every lazy read `server` keeps of memory `written` names, and order work on `writer`
/// after the copies of it, ahead of that work writing it.
///
/// Does nothing while `writer` records a graph: the recorded work writes nothing until a
/// replay, whose own write scope starts the reads then.
pub fn start_overwritten_reads<S: WriteScoped>(
    server: &mut S,
    writer: StreamId,
    written: &[BufferBinding],
) {
    if !written
        .iter()
        .any(|binding| binding.memory.descriptor().has_lazy_reads())
    {
        server.write_streams().lazy_reads().drop_retired();
        return;
    }
    if is_recording(server.capturing(writer)) {
        return;
    }
    let mut overwritten = server
        .write_streams()
        .lazy_reads()
        .take_overwritten(written);
    for kept in overwritten.reads {
        // The copy runs on the writer, behind the reader's work up to the read and none after
        // it, which may itself wait on the writer.
        let stream = match (&kept.made_at, kept.made_on == writer) {
            (_, true) => writer,
            (Some(made_at), false) => {
                if !made_at.is_reached() {
                    server.wait_for_fence(writer, made_at.as_ref());
                }
                writer
            }
            // A copy queued on a stream recording a graph would run at every replay instead.
            (None, false) => match is_recording(server.capturing(kept.made_on)) {
                true => writer,
                false => kept.made_on,
            },
        };
        overwritten
            .in_flight
            .extend(start(server, &kept.read, stream));
    }
    let in_flight = overwritten
        .in_flight
        .into_iter()
        .filter(|copy| copy.stream != writer);
    for copy in in_flight {
        match &copy.fence {
            Some(fence) => server.wait_for_fence(writer, fence.as_ref()),
            None => server.wait_before_write(writer, copy.stream),
        }
    }
}

/// Start `read` on `stream`, holding it with a fence past its copy when `server` gives one.
fn start<S: WriteScoped>(
    server: &mut S,
    read: &Arc<LazyRead>,
    stream: StreamId,
) -> Option<CopyInFlight> {
    let (sources, offsets) = read.to_held_sources();
    if sources.is_empty() {
        return None;
    }
    let started = server.read_before_write(sources, stream)?;
    let fence = server.fence_stream(stream);
    let reads = server.write_streams().lazy_reads();
    read.start(offsets, started, reads.retire(fence.clone()));
    reads.started(read, stream, fence.clone());
    Some(CopyInFlight { stream, fence })
}

/// Start every lazy read `server` keeps of `binding`'s memory, which kernels on other devices
/// may write from now on with nothing this device runs first.
///
/// # Errors
///
/// A refusal when a lazy read holds the memory and its stream records a graph, where the
/// read's copy would be recorded rather than run.
pub fn start_reads_before_sharing<S: WriteScoped>(
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

fn is_recording(capture: Option<&mut StreamCapture>) -> bool {
    capture.is_some_and(|capture| capture.is_recording())
}
