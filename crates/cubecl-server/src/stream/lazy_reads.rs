//! The lazy reads a device keeps, which work about to write their memory starts first.

use crate::memory_management::ManagedMemoryId;
use crate::server::{BufferBinding, LazyRead, ServerError};
use crate::stream::{FailureStore, StreamCapture, WriteScoped};
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use cubecl_environment::collections::HashMap;
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
}

/// What work about to write memory lazy reads hold has to do first.
#[derive(Debug, Default)]
pub struct Overwritten {
    /// The kept reads to start, each with the stream it was made on.
    pub reads: Vec<(Arc<LazyRead>, StreamId)>,
    /// The streams whose started copies of the memory may still be in flight.
    pub in_flight: Vec<StreamId>,
}

impl LazyReads {
    /// Keep `read`, made on `stream`, until work about to write its memory takes it.
    pub fn keep(&mut self, read: &Arc<LazyRead>, stream: StreamId) {
        read.keep();
        self.hold(read, stream);
    }

    /// Hold `read`, started on `stream`, until its copy resolves, so work writing its memory
    /// from another stream waits for it.
    pub fn started(&mut self, read: &Arc<LazyRead>, stream: StreamId) {
        self.hold(read, stream);
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
                let Some(read) = kept.read.upgrade().filter(|read| is_held(read)) else {
                    return false;
                };
                if !overlaps(&read.source().handle, binding) {
                    return true;
                }
                let is_kept = read.is_kept();
                match is_kept {
                    true => overwritten.reads.push((read, kept.stream)),
                    false => overwritten.in_flight.push(kept.stream),
                }
                !is_kept
            });
            if reads.is_empty() {
                self.kept.remove(&id);
            }
        }
        overwritten
    }

    fn hold(&mut self, read: &Arc<LazyRead>, stream: StreamId) {
        let reads = self
            .kept
            .entry(read.source().handle.memory.id())
            .or_default();
        reads.retain(|kept| kept.read.upgrade().is_some_and(|read| is_held(&read)));
        reads.push(KeptRead {
            read: Arc::downgrade(read),
            stream,
        });
        if self.kept.len() > 2 * self.swept_at.max(16) {
            self.sweep();
        }
    }

    /// Let go of the reads nothing holds, keeps or copies anymore, which no write would find.
    fn sweep(&mut self) {
        self.kept.retain(|_, reads| {
            reads.retain(|kept| kept.read.upgrade().is_some_and(|read| is_held(&read)));
            !reads.is_empty()
        });
        self.swept_at = self.kept.len();
    }
}

fn is_held(read: &LazyRead) -> bool {
    read.is_kept() || read.is_in_flight()
}

fn overlaps(read: &BufferBinding, written: &BufferBinding) -> bool {
    let (read, written) = (read.range(), written.range());
    read.start < written.end && written.start < read.end
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
    let streams = server.write_streams();
    streams.ensure_written(core::iter::once(&read.source().handle))?;
    streams.lazy_reads().keep(&read, stream);
    Ok(())
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
        let Some(started) = server.read_before_write(read.source().clone(), stream) else {
            continue;
        };
        read.start(started);
        server.write_streams().lazy_reads().started(&read, stream);
        overwritten.in_flight.push(stream);
    }
    overwritten.in_flight.sort_unstable();
    overwritten.in_flight.dedup();
    for reader in overwritten.in_flight {
        if reader != writer {
            server.wait_before_write(writer, reader);
        }
    }
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
