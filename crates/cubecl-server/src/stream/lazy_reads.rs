//! The lazy reads a device keeps, which work about to write their memory starts first.

use crate::memory_management::ManagedMemoryId;
use crate::server::{BufferBinding, LazyRead, ServerError};
use crate::stream::{FailureStore, StreamCapture, WriteScoped};
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use cubecl_environment::collections::HashMap;
use cubecl_environment::stream::StreamId;

/// The lazy reads a device keeps, until work about to write their memory starts them or their
/// bytes let them go.
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
    /// The stream the read was made on, which the work it was made after ran on.
    stream: StreamId,
}

/// A lazy read work is about to write the memory of.
#[derive(Debug)]
pub struct OverwrittenRead {
    /// The read.
    pub read: Arc<LazyRead>,
    /// The stream it was made on.
    pub stream: StreamId,
}

impl LazyReads {
    /// Keep `read`, made on `stream`, until work about to write its memory takes it.
    pub fn keep(&mut self, read: &Arc<LazyRead>, stream: StreamId) {
        read.keep();
        self.kept
            .entry(read.source().handle.memory.id())
            .or_default()
            .push(KeptRead {
                read: Arc::downgrade(read),
                stream,
            });
        if self.kept.len() > 2 * self.swept_at.max(16) {
            self.sweep();
        }
    }

    /// The kept reads whose bytes `written` overlaps, which are no longer kept: the work about
    /// to write them starts them.
    pub fn take_overwritten(&mut self, written: &[BufferBinding]) -> Vec<OverwrittenRead> {
        let mut taken = Vec::new();
        let written = written
            .iter()
            .filter(|binding| binding.memory.descriptor().has_lazy_reads());
        for binding in written {
            let id = binding.memory.id();
            let Some(reads) = self.kept.get_mut(&id) else {
                continue;
            };
            reads.retain(|kept| {
                let Some(read) = kept.read.upgrade().filter(|read| read.is_kept()) else {
                    return false;
                };
                if !overlaps(&read.source().handle, binding) {
                    return true;
                }
                taken.push(OverwrittenRead {
                    read,
                    stream: kept.stream,
                });
                false
            });
            if reads.is_empty() {
                self.kept.remove(&id);
            }
        }
        taken
    }

    /// Let go of the reads nothing holds or keeps anymore, which no write would find to take.
    fn sweep(&mut self) {
        self.kept.retain(|_, reads| {
            reads.retain(|kept| kept.read.upgrade().is_some_and(|read| read.is_kept()));
            !reads.is_empty()
        });
        self.swept_at = self.kept.len();
    }
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
    if server
        .capturing(stream)
        .is_some_and(|capture| capture.is_recording())
    {
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

/// Start every lazy read `server` keeps of memory `written` names, ahead of work on `writer`
/// that writes it.
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
    for overwritten in server
        .write_streams()
        .lazy_reads()
        .take_overwritten(written)
    {
        // A copy queued on a stream recording a graph would run at every replay instead.
        let stream = match is_recording(server.capturing(overwritten.stream)) {
            true => writer,
            false => overwritten.stream,
        };
        let source = overwritten.read.source().clone();
        if let Some(started) = server.read_before_write(source, stream, writer) {
            overwritten.read.start(started);
        }
    }
}

fn is_recording(capture: Option<&mut StreamCapture>) -> bool {
    capture.is_some_and(|capture| capture.is_recording())
}
