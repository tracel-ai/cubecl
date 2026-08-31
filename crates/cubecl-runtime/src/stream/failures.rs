//! The device's failure store, and the surface every multi-stream driver
//! gives over it.
//!
//! Two drivers exist — the event-ordered [`MultiStream`](super::MultiStream)
//! and the [`SchedulerMultiStream`](super::scheduler::SchedulerMultiStream) —
//! and every rule about taint is the same on both. So the state lives in one
//! type here, the operations are default methods on one trait, and a driver
//! supplies only the thing that differs: where its pool and its store are.
//! Written twice, the two would drift on the first rule either one gained.

use crate::id::KernelId;
use crate::logging::ServerLogger;
use crate::memory_management::{ErrorGraph, FailureId, Skipped};
use crate::server::{BufferBinding, ServerError};
use crate::stream::{ReadFailure, StreamFactory, StreamMemory, StreamPool, base};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Every failure a device is still holding, plus the little a write scope
/// needs around them.
///
/// One store per device rather than per stream, because a launch failing on
/// one stream can claim a slice owned by another and both have to point at
/// the same thing.
///
/// There is no device-wide failure beside the graph. A failure belongs to the
/// memory it left unwritten, and work that shares no buffer with it is
/// unaffected — which is the whole point, and a slot that failed the next
/// flush regardless of what it was flushing was the one thing that broke it.
#[derive(Debug)]
pub struct Failures {
    graph: ErrorGraph,
    /// The vector a write scope stages its write set in, pooled here so a
    /// launch allocates nothing for it.
    scratch: Vec<BufferBinding>,
    logger: Arc<ServerLogger>,
}

impl Failures {
    /// An empty store for a device that has failed at nothing yet.
    pub fn new(logger: Arc<ServerLogger>) -> Self {
        Self {
            graph: ErrorGraph::default(),
            scratch: Vec::new(),
            logger,
        }
    }

    /// What the failure ids carried by this device's allocations mean.
    pub fn graph(&self) -> &ErrorGraph {
        &self.graph
    }

    /// [`graph`](Self::graph) mutably, handed down to every reserve, bind and
    /// cleanup — those are where slices shed the failures they carry.
    pub fn graph_mut(&mut self) -> &mut ErrorGraph {
        &mut self.graph
    }
}

/// A multi-stream driver that owns a device's [`Failures`].
///
/// Implementing [`split`](Self::split) and [`parts`](Self::parts) buys the
/// whole taint surface below, and the write scope's hooks with it. Whether a
/// buffer can be trusted lives on its allocation, so every operation here is
/// the same two steps: resolve each binding to the stream that allocated it,
/// and tell that stream's memory what happened.
pub trait FailureStore {
    /// The factory the driver's [`StreamPool`] was built from. Its streams
    /// expose the memory the taint is recorded on.
    type Factory: StreamFactory<Stream: StreamMemory>;

    /// The pool and the store, split-borrowed: nearly every operation here
    /// reaches the allocations through the pool while mutating the store.
    fn split(&mut self) -> (&mut StreamPool<Self::Factory>, &mut Failures);

    /// [`split`](Self::split) for the read-only questions.
    fn parts(&self) -> (&StreamPool<Self::Factory>, &Failures);

    /// Fails when the buffers `handles` name carry a failure, with the errors
    /// of the work that was supposed to write them.
    ///
    /// A read is only as good as the work that wrote the buffer: a launch that
    /// failed never wrote it, so copying its bytes out hands back whatever was
    /// in memory before. Whether that happened is a field on the allocation,
    /// so the question is answered by the slice each binding resolves to — a
    /// lookup the read was going to do anyway — and by nobody's queue.
    ///
    /// # Errors
    ///
    /// [`ServerError::Several`] naming every failure one of these buffers
    /// carries, each failure once however many buffers carry it. The caller
    /// has nothing to retry — the bytes are gone — so the error is the answer
    /// to the read, not a hint to try again.
    fn ensure_written<'a>(
        &self,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError> {
        let (pool, failures) = self.parts();
        failures.graph.reports(handles.filter_map(|handle| {
            let failure = pool.try_get(&handle.stream)?.failure(handle)?;
            Some((failure, handle.memory.id()))
        }))
    }

    /// The failure claiming bytes any of `reads` names, with its error — the
    /// check a launch makes before it runs.
    ///
    /// A launch whose input cannot be trusted does not run: a buffer holding
    /// garbage can be read as a dynamic cube count or as indices in a gather,
    /// so running would risk dispatching an absurd grid or scattering into
    /// memory that carried no failure at all. Skipping costs the same lookup,
    /// because the inputs have to be read either way to decide anything.
    fn read_failure<'a>(
        &self,
        mut reads: impl Iterator<Item = &'a BufferBinding>,
    ) -> Option<ReadFailure> {
        let (pool, failures) = self.parts();
        reads.find_map(|handle| {
            let failure = pool.try_get(&handle.stream)?.failure(handle)?;
            Some(ReadFailure {
                failure,
                needed: handle.memory.id(),
                error: failures.graph.error(failure)?.clone(),
            })
        })
    }

    /// Taint every allocation in `written` with `error`: the work that was
    /// going to write those buffers did not run, so a read of any of them
    /// fails on this failure until something writes them again.
    ///
    /// Each binding is resolved to the manager of the stream it was created
    /// on, which may not be the stream that failed — that is the point: the
    /// fact lands on the memory, wherever it lives.
    fn taint<'a>(&mut self, error: ServerError, written: impl Iterator<Item = &'a BufferBinding>) {
        let (pool, failures) = self.split();
        base::taint(pool, error, written, &mut failures.graph);
    }

    /// Release the failure on every allocation in `written`: work that writes
    /// them has been enqueued, so a read of one is no longer reading bytes
    /// nothing wrote.
    fn written<'a>(&mut self, written: impl Iterator<Item = &'a BufferBinding>) {
        let (pool, failures) = self.split();
        base::written(pool, written, &mut failures.graph);
    }

    /// A skipped launch's outputs take the failure that stopped it: nothing
    /// wrote them, exactly as if the launch had failed, and the claim names
    /// the root cause rather than minting a new one. The skip is recorded on
    /// the failure, so a read of anything downstream can name the path back
    /// to the root.
    ///
    /// Takes the write set by value and hands it back to the pool, the same
    /// contract [`exit_write`](Self::exit_write) has, because a skip is the
    /// other way a scope ends: a loop carrying a tainted buffer forward skips
    /// on every iteration — the most frequent event in this whole design — and
    /// a set the skip path dropped would allocate a fresh one every time.
    fn propagate(
        &mut self,
        found: &ReadFailure,
        kernel: KernelId,
        mut written: Vec<BufferBinding>,
    ) {
        let (pool, failures) = self.split();
        failures.graph.skipped(
            found.failure,
            Skipped {
                kernel,
                needed: found.needed,
                produced: written.iter().map(|handle| handle.memory.id()).collect(),
            },
        );
        base::taint_with(pool, found.failure, written.iter(), &mut failures.graph);
        written.clear();
        failures.scratch = written;
    }

    /// An empty write set, pooled here so a launch allocates nothing for it.
    /// [`exit_write`](Self::exit_write) hands it back.
    fn write_set(&mut self) -> Vec<BufferBinding> {
        let (_, failures) = self.split();
        core::mem::take(&mut failures.scratch)
    }

    /// Enter a write scope over `written`: taint every buffer the work is
    /// going to write with a provisional failure, minted here because the real
    /// one does not exist yet.
    ///
    /// The default this sets is tainted unless proven written — the opposite
    /// of clearing on success and hoping every failure path remembered to
    /// taint. A body that returns early, or panics before
    /// [`exit_write`](Self::exit_write) runs, leaves the write set carrying
    /// this failure, so a read of one of its buffers fails loudly instead of
    /// returning bytes nothing wrote.
    ///
    /// An empty write set — a dry run, a launch writing nothing — claims
    /// nothing and mints nothing.
    fn enter_write(&mut self, written: &[BufferBinding]) -> Option<FailureId> {
        if written.is_empty() {
            return None;
        }
        let (pool, failures) = self.split();
        // Payload-free on purpose: this node is minted and dropped again on
        // every launch that succeeds, so it may not cost a formatted string
        // or a stack walk. See [`ServerError::TornDown`].
        let provisional = failures.graph.insert(ServerError::TornDown);
        base::taint_with(pool, provisional, written.iter(), &mut failures.graph);
        Some(provisional)
    }

    /// Settle the scope entered over `written`: release the provisional
    /// failure when the work was enqueued, and swap the real error in for it
    /// when the work was not. The taint is the whole answer — a read of one
    /// of these buffers fails on it, whoever asks — and the error is logged
    /// here, the backstop for the failure nobody ever reads. The staged
    /// vector goes back to the pool either way.
    fn exit_write(
        &mut self,
        provisional: Option<FailureId>,
        mut written: Vec<BufferBinding>,
        error: Option<&ServerError>,
    ) {
        match error {
            None => self.written(written.iter()),
            Some(error) => {
                let (_, failures) = self.split();
                failures.logger.log_failure(error);
                if let Some(provisional) = provisional {
                    failures.graph.replace(provisional, error.clone());
                }
            }
        }
        let (_, failures) = self.split();
        // Covers the failure that tainted nothing — every binding resolving to
        // a slot no stream ever initialized — and costs one lookup otherwise.
        if let Some(provisional) = provisional {
            failures.graph.prune(provisional);
        }
        written.clear();
        failures.scratch = written;
    }
}
