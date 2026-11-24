//! The lazy reads a device keeps, which work about to write their memory starts first.

use crate::device_events::{Event, EventApi};
use crate::memory_management::ManagedMemoryId;
use crate::server::{BufferBinding, LazyRead, ServerError};
use crate::stream::{FailureStore, StreamCapture, WriteScoped};
use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::collections::HashMap;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;

/// The lazy reads a device keeps, until work about to write their memory starts them and their
/// copies resolve, or their bytes let them go.
#[derive(Debug, Default)]
pub struct LazyReads {
    /// Keyed usage: the reads of each allocation, looked up by the allocation a write names.
    kept: HashMap<ManagedMemoryId, Vec<KeptRead>>,
    /// How many allocations `kept` named when it last let go of the reads nothing holds.
    swept_at: usize,
}

#[derive(Debug)]
struct KeptRead {
    read: Weak<LazyRead>,
    /// The stream the read was made on until it is started, then the stream its copy runs on.
    stream: StreamId,
    /// Past the copy once the read is started, when its server gives one.
    fence: Option<Arc<dyn CopyFence>>,
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

/// The point on a stream just past a lazy read's copy, which work writing its memory from
/// another stream waits on rather than on all the work queued there since.
pub trait CopyFence: core::any::Any + Send + Sync + core::fmt::Debug {
    /// Whether the device has finished the copy.
    fn is_reached(&self) -> bool;
}

impl<A: EventApi> CopyFence for Event<A>
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
    pub fence: Option<Arc<dyn CopyFence>>,
}

/// What work about to write memory lazy reads hold has to do first.
#[derive(Debug, Default)]
pub struct Overwritten {
    /// The kept reads to start, each with the stream it was made on.
    pub reads: Vec<(Arc<LazyRead>, StreamId)>,
    /// The started copies of the memory that may still be running.
    pub in_flight: Vec<CopyInFlight>,
}

impl LazyReads {
    /// Keep `read`, made on `stream`, until work about to write its memory takes it.
    pub fn keep(&mut self, read: &Arc<LazyRead>, stream: StreamId) {
        read.keep();
        self.hold(read, stream, None, None);
    }

    /// Hold `copy`, of `accessed`, bytes of `read` an access reads on `stream` while it stays
    /// kept, so work writing them waits for it, on `fence` when there is one.
    fn accessed(
        &mut self,
        read: &Arc<LazyRead>,
        accessed: Range<u64>,
        copy: &Arc<AccessCopy>,
        stream: StreamId,
        fence: Option<Arc<dyn CopyFence>>,
    ) {
        self.hold(read, stream, fence, Some((accessed, Arc::downgrade(copy))));
    }

    /// Hold `read`, started on `stream`, until its copy is done, so work writing its memory
    /// from another stream waits for it, on `fence` when there is one.
    pub fn started(
        &mut self,
        read: &Arc<LazyRead>,
        stream: StreamId,
        fence: Option<Arc<dyn CopyFence>>,
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
                kept.fence = fence;
            }
            None => self.hold(read, stream, fence, None),
        }
    }

    /// What work about to write `written` has to do first: the kept reads whose bytes it
    /// overlaps, which are no longer kept, and the streams copies of them are in flight on.
    pub fn take_overwritten(&mut self, written: &[BufferBinding]) -> Overwritten {
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
                    true => overwritten.reads.push((read, kept.stream)),
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
        fence: Option<Arc<dyn CopyFence>>,
        accessed: Option<(Range<u64>, Weak<AccessCopy>)>,
    ) {
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
            fence,
            accessed,
        });
        if self.kept.len() > 2 * self.swept_at.max(16) {
            self.sweep();
        }
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
    let streams = server.write_streams();
    streams.ensure_written(core::iter::once(&source.handle))?;
    streams.lazy_reads().keep(&read, stream);
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
    let fence = server.fence_copy(stream);
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
        return;
    }
    if is_recording(server.capturing(writer)) {
        return;
    }
    let mut overwritten = server
        .write_streams()
        .lazy_reads()
        .take_overwritten(written);
    for (read, made_on) in overwritten.reads {
        // A copy queued on a stream recording a graph would run at every replay instead.
        let stream = match is_recording(server.capturing(made_on)) {
            true => writer,
            false => made_on,
        };
        overwritten.in_flight.extend(start(server, &read, stream));
    }
    let in_flight = overwritten
        .in_flight
        .into_iter()
        .filter(|copy| copy.stream != writer);
    for copy in in_flight {
        match &copy.fence {
            Some(fence) => server.wait_for_copy(writer, fence.as_ref()),
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
    let (source, offset) = read.to_held_source();
    let started = server.read_before_write(source, stream)?;
    read.start(offset, started);
    let fence = server.fence_copy(stream);
    server
        .write_streams()
        .lazy_reads()
        .started(read, stream, fence.clone());
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
