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
use crate::stream::{ReadFailure, StreamFactory, StreamMemory, StreamPool, WriteStreams};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Every failure a device is still holding, plus the little a write scope
/// needs around them.
///
/// One store per device rather than per stream, because a launch failing on
/// one stream can taint a slice owned by another and both have to point at
/// the same thing.
#[derive(Debug)]
pub struct Failures {
    graph: ErrorGraph,
    /// The vector a write scope stages its write set in, pooled here so a
    /// launch allocates nothing for it.
    scratch: Vec<BufferBinding>,
    /// The one failure that is a device's rather than any buffer's: a fault
    /// the driver reports against the context — a failed command buffer, a
    /// validation canary, a synchronize that failed on a path with no caller
    /// to hand it to. Reported by the next flush or sync, which is the only
    /// thing left that nobody else can tell the caller.
    fault: Option<ServerError>,
    logger: Arc<ServerLogger>,
}

impl Failures {
    /// An empty store for a device that has failed at nothing yet.
    pub fn new(logger: Arc<ServerLogger>) -> Self {
        Self {
            graph: ErrorGraph::default(),
            scratch: Vec::new(),
            fault: None,
            logger,
        }
    }

    /// What the failure ids carried by this device's allocations mean, handed
    /// down to every reserve, bind and cleanup — those are where slices shed
    /// the failures they carry.
    pub fn graph_mut(&mut self) -> &mut ErrorGraph {
        &mut self.graph
    }
}

/// A multi-stream driver that owns a device's [`Failures`].
///
/// Implementing [`split`](Self::split) and [`parts`](Self::parts) buys the
/// whole taint surface below, and [`WriteStreams`] with it.
pub trait FailureStore {
    /// The factory the driver's [`StreamPool`] was built from.
    type Factory: StreamFactory;

    /// The pool and the store, split-borrowed: nearly every operation here
    /// reaches the allocations through the pool while mutating the store.
    fn split(&mut self) -> (&mut StreamPool<Self::Factory>, &mut Failures);

    /// [`split`](Self::split) for the read-only questions.
    fn parts(&self) -> (&StreamPool<Self::Factory>, &Failures);

    /// Fails when the buffers `handles` name carry a failure — see
    /// [`StreamPool::ensure_written`].
    ///
    /// # Errors
    ///
    /// [`ServerError::Several`] naming the work that was supposed to
    /// write them and failed.
    fn ensure_written<'a>(
        &self,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError>
    where
        <Self::Factory as StreamFactory>::Stream: StreamMemory,
    {
        let (pool, failures) = self.parts();
        pool.ensure_written(&failures.graph, handles)
    }

    /// The failure claiming bytes any of `reads` names — see
    /// [`StreamPool::read_failure`].
    fn read_failure<'a>(
        &self,
        reads: impl Iterator<Item = &'a BufferBinding>,
    ) -> Option<ReadFailure>
    where
        <Self::Factory as StreamFactory>::Stream: StreamMemory,
    {
        let (pool, failures) = self.parts();
        pool.read_failure(&failures.graph, reads)
    }

    /// Taint every allocation in `written` with `error` — see
    /// [`StreamPool::taint`].
    fn taint<'a>(&mut self, error: ServerError, written: impl Iterator<Item = &'a BufferBinding>)
    where
        <Self::Factory as StreamFactory>::Stream: StreamMemory,
    {
        let (pool, failures) = self.split();
        pool.taint(error, written, &mut failures.graph);
    }

    /// Release the failure on every allocation in `written` — see
    /// [`StreamPool::written`].
    fn written<'a>(&mut self, written: impl Iterator<Item = &'a BufferBinding>)
    where
        <Self::Factory as StreamFactory>::Stream: StreamMemory,
    {
        let (pool, failures) = self.split();
        pool.written(written, &mut failures.graph);
    }

    /// A skipped launch's outputs take the failure that stopped it: nothing
    /// wrote them, exactly as if the launch had failed, and the claim names
    /// the root cause rather than minting a new one. The skip is recorded on
    /// the failure, so a read of anything downstream can name the path back
    /// to the root.
    ///
    /// The write set is staged in the pooled scratch vector, because a loop
    /// carrying a tainted buffer forward skips on every iteration — the most
    /// frequent event in this whole design.
    fn propagate<'a>(
        &mut self,
        found: &ReadFailure,
        kernel: KernelId,
        written: impl Iterator<Item = &'a BufferBinding>,
    ) where
        <Self::Factory as StreamFactory>::Stream: StreamMemory,
    {
        let (pool, failures) = self.split();
        let mut staged = core::mem::take(&mut failures.scratch);
        staged.extend(written.cloned());
        failures.graph.skipped(
            found.failure,
            Skipped {
                kernel,
                needed: found.needed,
                produced: staged.iter().map(|handle| handle.memory.id()).collect(),
            },
        );
        pool.taint_with(found.failure, staged.iter(), &mut failures.graph);
        staged.clear();
        failures.scratch = staged;
    }

    /// Record a device fault — the failure that is the context's rather than
    /// any buffer's. The first fault wins: it is the one that broke the
    /// context, and it is logged either way.
    fn fault(&mut self, error: ServerError) {
        let (_, failures) = self.split();
        failures.logger.log_failure(&error);
        if failures.fault.is_none() {
            failures.fault = Some(error);
        }
    }

    /// The device fault owed to the next flush or sync, taken — reported
    /// once, like the queue it replaces.
    fn take_fault(&mut self) -> Option<ServerError> {
        let (_, failures) = self.split();
        failures.fault.take()
    }
}

impl<T: FailureStore> WriteStreams for T
where
    <T::Factory as StreamFactory>::Stream: StreamMemory,
{
    fn stage(&mut self) -> Vec<BufferBinding> {
        let (_, failures) = self.split();
        core::mem::take(&mut failures.scratch)
    }

    fn enter(&mut self, written: &[BufferBinding]) -> Option<FailureId> {
        let (pool, failures) = self.split();
        pool.enter_write(written, &mut failures.graph)
    }

    fn exit(
        &mut self,
        provisional: Option<FailureId>,
        mut written: Vec<BufferBinding>,
        error: Option<&ServerError>,
    ) {
        let (pool, failures) = self.split();
        if let Some(error) = error {
            // The backstop of the lazy model: the taint reports through every
            // read, and this line covers the failure nobody ever reads.
            failures.logger.log_failure(error);
        }
        pool.exit_write(provisional, &written, error, &mut failures.graph);
        written.clear();
        failures.scratch = written;
    }
}
