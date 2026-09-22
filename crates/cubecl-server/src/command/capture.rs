//! Recording a stream's launches into a replayable graph.
//!
//! The driver's rules are what shapes this. A stream in capture mode records
//! every launch issued on it, so a window that opened has to be closed even
//! when what it recorded is worthless. Nothing may allocate inside the window,
//! so a warmup run leaves the pools holding what the recording asks for, and
//! the pages the recording touched are guarded once it closes. And a host
//! sync would abort the recording, so the fenced flushes the execution path
//! would otherwise run are deferred across it.
//!
//! All of that is the same whichever driver is underneath. What is not is
//! [`GraphDriver`]: opening the recording, closing it into an executable,
//! staging that executable, and replaying it.

use super::{Command, DeviceStream, Driver};
use crate::id::GraphId;
use crate::memory_management::PageGuard;
use crate::server::{BufferBinding, ServerError};
use alloc::format;
use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_environment::stream::StreamId;

/// A driver that can record a stream's launches into a replayable graph.
pub trait GraphDriver: Driver {
    /// An instantiated graph, released when it drops.
    ///
    /// Dropping is how it is destroyed, so no path can hand back a graph and
    /// leak the executable behind it — including the ones that instantiate
    /// successfully and then find the window was abandoned.
    type Executable;

    /// Put `stream` into recording mode: from here the driver records every
    /// launch issued on it.
    ///
    /// # Errors
    ///
    /// The driver's refusal to begin recording. The caller restores the stream
    /// to what it was, so the whole sequence can be retried.
    fn begin(stream: &mut Self::Stream) -> Result<(), ServerError>;

    /// Close the recording on `stream` and instantiate what it recorded.
    ///
    /// `doomed` is a recording already known not to become a graph — work
    /// inside the window failed or was skipped, so an operation is missing.
    /// The driver capture is closed either way: a stream left in capture mode
    /// records every launch that follows it.
    ///
    /// A recording that allocated is refused here too. A graph owning memory
    /// nodes allocates on launch and never frees, so the driver rejects every
    /// relaunch while the first quietly succeeds.
    ///
    /// # Errors
    ///
    /// Whatever stopped the recording from becoming a graph, `doomed`
    /// included.
    fn instantiate(
        stream: &mut Self::Stream,
        doomed: Option<ServerError>,
    ) -> Result<Self::Executable, ServerError>;

    /// Pre-stage `exec` so the first replay does not pay the upload cost.
    ///
    /// Non-fatal by contract: a replay uploads on demand if this does nothing.
    fn upload(exec: &Self::Executable, stream: &mut Self::Stream);

    /// Enqueue `exec`'s recorded sequence on `stream`.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the replay.
    fn replay(exec: &Self::Executable, stream: &mut Self::Stream) -> Result<(), ServerError>;
}

/// An instantiated graph, and everything its window pinned for it.
///
/// Owned by [`Captures`] and referenced by [`GraphId`]; the executable never
/// leaves the server actor, which serializes access, so it is only ever
/// touched on the one thread allowed to. The client references the graph by id
/// and, on the last drop, asks the actor to release it — the server syncs the
/// stream before [`Captures::destroy`], so the executable is never destroyed
/// while a replay is still running.
pub struct Graph<D: GraphDriver> {
    exec: D::Executable,
    /// Every page the recorded launches were given memory on, guarded for the
    /// graph's lifetime. A replay re-runs the recorded kernels against these
    /// exact device pointers, so nothing on those pages may move or be handed
    /// to a later allocation the replay would overwrite.
    _guards: Vec<PageGuard>,
    /// The host memory the graph's recorded copies read from, alive for its
    /// lifetime for the same reason: a memcpy node keeps the raw host
    /// pointer, and every replay reads through it again.
    _retained_host: Vec<Bytes>,
    /// The buffers the recorded launches write, deduplicated. A replay that
    /// fails to enqueue runs none of those launches, so it leaves every one of
    /// these as it was — claiming them is what makes a later read of one fail,
    /// whichever stream asks.
    written: Vec<BufferBinding>,
}

/// A capture window that closed without a graph to hand back.
///
/// Not an error on its own, because the recorded launches never ran and now
/// never will: the memory they would have written is left exactly as it was,
/// and someone has to claim it before a later read of one of those buffers
/// returns bytes nothing wrote. The window cannot do that itself — it has no
/// failure store — so it says what to claim and for which error.
pub struct Refused {
    /// Why no graph came out.
    pub error: ServerError,
    /// The memory the recording's launches would have written.
    pub written: Vec<BufferBinding>,
}

/// The graphs this device has instantiated, keyed by the [`GraphId`] handed to
/// the client.
///
/// Referencing a graph by id is what keeps the executable inside the server:
/// nothing across the actor boundary ever holds one.
pub struct Captures<D: GraphDriver> {
    graphs: HashMap<GraphId, Graph<D>>,
}

impl<D: GraphDriver> core::fmt::Debug for Captures<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The executables are opaque driver handles, so the count is the whole
        // of what there is to say.
        f.debug_struct("Captures")
            .field("graphs", &self.graphs.len())
            .finish()
    }
}

impl<D: GraphDriver> Default for Captures<D> {
    fn default() -> Self {
        Self {
            graphs: HashMap::default(),
        }
    }
}

impl<D: GraphDriver> Captures<D> {
    /// Register a freshly instantiated graph under the id its capture was
    /// given.
    pub fn insert(&mut self, id: GraphId, graph: Graph<D>) {
        self.graphs.insert(id, graph);
    }

    /// Whether `id` still names a live graph.
    pub fn contains(&self, id: GraphId) -> bool {
        self.graphs.contains_key(&id)
    }

    /// Add the buffers `id`'s recorded launches write to `written`.
    ///
    /// Extends rather than answers with a vector of its own, because the
    /// caller is filling a pooled write set and a replay should allocate for
    /// it no more than a launch does.
    ///
    /// An unknown id adds nothing, which is the honest answer rather than a
    /// missing one: a graph that is gone took the record of which buffers went
    /// with it, and a replay of it writes nothing.
    pub fn extend_written(&self, id: GraphId, written: &mut Vec<BufferBinding>) {
        if let Some(graph) = self.graphs.get(&id) {
            written.extend(graph.written.iter().cloned());
        }
    }

    /// Enqueue `id`'s recorded sequence on `stream`.
    ///
    /// The stream's existing errors are ignored — they surface on the next
    /// sync — so a replay only ever adds its own.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generic`] when `id` names no live graph, which the
    /// caller hands straight back: nothing was enqueued and nothing is stale.
    pub fn replay(&self, id: GraphId, stream: &mut D::Stream) -> Result<(), ServerError> {
        let graph = self.graphs.get(&id).ok_or_else(|| ServerError::Generic {
            reason: "replay was given an unknown or already-destroyed graph".into(),
            backtrace: BackTrace::capture(),
        })?;
        D::replay(&graph.exec, stream)
    }

    /// Drop the executable `id` names and release what it held: the buffers it
    /// pinned go with it, and the info-cache entries no other live graph still
    /// pins are freed. A no-op for an unknown id, so a double release is one.
    ///
    /// The caller syncs `stream` first — a replay enqueued against this
    /// executable may still be running.
    pub fn destroy(&mut self, id: GraphId, stream: &mut D::Stream) {
        self.graphs.remove(&id);
        stream.info_cache().graph_release(id);
    }
}

/// A capture window on one stream: warming up before it opens, opening it,
/// and instantiating what it recorded.
///
/// The three steps are ordered and the stream refuses them out of order (see
/// [`StreamCapture`](crate::stream::StreamCapture)); this type is where each
/// one's shared half lives.
pub struct Window<'c, 'a, D: GraphDriver> {
    command: &'c mut Command<'a, D>,
}

impl<'c, 'a, D: GraphDriver> Window<'c, 'a, D> {
    /// The capture window on the stream `command` is issued on.
    pub fn on(command: &'c mut Command<'a, D>) -> Self {
        Self { command }
    }

    fn stream(&mut self) -> &mut D::Stream {
        self.command.stream()
    }

    /// Start the warmup run, before the window opens.
    ///
    /// The outdated pools are emptied first: every page the recording touches
    /// is guarded for the graph's life, and a guarded page can no longer be
    /// relocated off.
    ///
    /// # Errors
    ///
    /// The stream's refusal, when it is not in a state a window can open from.
    pub fn prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.stream().capturing().prepare(stream_id)?;
        self.command.relocate();
        Ok(())
    }

    /// Open the window: from here the driver records every launch issued on
    /// this stream until the window closes.
    ///
    /// # Errors
    ///
    /// The driver's refusal to begin recording, with the stream left as it was
    /// found — capture state back to none — so the whole prepare-then-open
    /// sequence can be retried.
    pub fn begin(&mut self) -> Result<(), ServerError> {
        // Rejected before the reclaim below runs: a drop-queue flush issued on
        // a stream that is already recording would abort its live capture.
        // Opening releases the staging the warmup run primed.
        self.stream().capturing().begin()?;
        // Reclaim deferred frees before the window opens: warmup's pinned
        // staging buffers (and any other drop-queued slices) sit in the drop
        // queue until drained, so without this the recorded run finds no free
        // staging slice to reuse.
        let stream = self.stream();
        let signal = stream.signal();
        stream.drop_queue().drain(|| D::Stream::fence(signal));

        if let Err(err) = D::begin(stream) {
            stream.info_cache().capture_discard();
            stream.capturing().abort();
            return Err(err);
        }
        // Recording now: fenced drop-queue flushes on the execution path are
        // suppressed for as long as the window is open, since a host sync
        // would abort it. The deferred buffers are reclaimed when it closes.
        Ok(())
    }

    /// Close the window and instantiate what it recorded into a graph
    /// registered under `id`.
    ///
    /// The window leaves capture mode first, so none of the paths below can
    /// wedge the stream in it — they re-enable the deferred fenced flushes on
    /// the way out. A window the caller does not own is closed and torn down
    /// all the same, since nobody else is coming back to close it; only its
    /// owner gets a graph out of it.
    ///
    /// # Errors
    ///
    /// [`Refused`], which every failure here becomes — the driver's refusal to
    /// close or instantiate, a recording an unreadable input left incomplete,
    /// an allocation inside the window, or no window at all. It carries the
    /// memory the caller now has to claim, empty when there was nothing to
    /// close.
    pub fn instantiate(&mut self, stream_id: StreamId, id: GraphId) -> Result<Graph<D>, Refused> {
        let outcome = match self.stream().capturing().end(stream_id) {
            Ok(outcome) => outcome,
            // No window to close, so nothing was recorded and the drained set
            // is empty — drained rather than assumed so the claim is right
            // even if that ever stops being true.
            Err(error) => {
                let stream = self.stream();
                let written = stream.capturing().take_recorded();
                // A capture prepared but never opened still holds the stream,
                // and a `graph_prepare` retry is refused while it does.
                // Closing is the only call the caller has left — a warmup that
                // failed never reaches `begin` — so a close from `Prepare`
                // unwinds it, as `begin` does when the driver refuses to open.
                if stream.capturing().is_active() {
                    stream.info_cache().capture_discard();
                    stream.capturing().abort();
                }
                return Err(Refused { error, written });
            }
        };
        // Work inside the window failed or was skipped, so the recording is
        // missing an operation and must not seal; the driver capture still
        // has to be closed either way.
        let doomed = self.stream().capturing().take_failure().map(|reason| {
            ServerError::graph_state(format!(
                "an operation inside the capture window failed or was skipped, so the \
                 recording is missing an operation and cannot seal: {reason}"
            ))
        });
        let exec = D::instantiate(self.stream(), doomed.clone());
        // Guard every page the window touched, on whichever stream owns it, so
        // nothing on them moves or is reused for the graph's lifetime. On
        // failure the guards drop with the refusal.
        let touched = self.stream().capturing().take_touched();
        let guards = self.command.guard_pages(&touched);
        let stream = self.stream();
        // The host bytes the recorded copies read from, whatever their kind —
        // a pool slice, a user buffer, a heap fallback. The nodes keep their
        // raw pointers, so they live with the graph; on a window that seals
        // no graph they drop here, since its copies never ran and never will.
        let retained_host = stream.capturing().take_retained_host();
        // Reclaim the buffers dropped while the window was open, whose fenced
        // flushes were deferred for as long as it was.
        let signal = stream.signal();
        stream.drop_queue().drain(|| D::Stream::fence(signal));
        // The memory the recorded launches write. A recording that becomes a
        // graph answers for it on a failed replay; one that does not is
        // answered for by the caller, since those launches never ran and now
        // never will.
        let written = stream.capturing().take_recorded();
        // An abandoned window has no graph to hand back: whatever was
        // instantiated drops here, and the report carries along whatever had
        // already doomed the recording so the caller sees both reasons.
        let exec = match outcome.is_abandoned() {
            false => exec,
            true => Err(outcome.abandoned_error(stream_id, doomed)),
        };
        match exec {
            Ok(exec) => {
                // Seal the info-cache entries this window pinned under the
                // graph's id, so destroying it can release them later.
                stream.info_cache().capture_commit(id);
                D::upload(&exec, stream);
                Ok(Graph {
                    exec,
                    _guards: guards,
                    _retained_host: retained_host,
                    written,
                })
            }
            Err(error) => {
                // Unpin the entries this window pinned — they stay as ordinary
                // cached values — and drop the guards.
                stream.info_cache().capture_discard();
                Err(Refused { error, written })
            }
        }
    }
}
