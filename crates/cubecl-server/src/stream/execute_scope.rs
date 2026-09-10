//! One scope around a unit of device work.
//!
//! Without the scope, every backend keeps the same bookkeeping by hand:
//! decide whether an input can be trusted, gather what the work writes, claim
//! it on every failure path, release it on the success path, and hope no
//! early return forgot. The scope makes forgetting loud instead of silent.
//!
//! A scope is opened one of two ways and the choice is made once, in the
//! constructor, which is why the two can never interleave:
//!
//! - a launch whose inputs all read cleanly, or work that reads nothing it
//!   must trust, **enters**: the write set is claimed by a provisional
//!   failure, and leaving either releases it or swaps the real error in;
//! - a launch whose input carries a failure **skips**: the write set is
//!   pointed at that same failure instead, so a read downstream names the
//!   original cause rather than a new one, and the body never runs.
//!
//! Entering on a skip would be wrong rather than merely wasteful — the
//! provisional would be minted, overwritten by the propagated failure, and
//! never pruned, because the exit that prunes it is on the path that does not
//! run. Nothing has to remember that, because a scope is one or the other
//! before it exists.
//!
//! The body is a closure and the scope is not a guard value, because a guard
//! enforces nothing here: `#[must_use]` says nothing about a bound value on a
//! path that returns early, and a `Drop` implementation cannot reach the
//! failure store and must not assert during an unwind. A path that never
//! reaches the exit — a panic mid-launch above all — leaves the write set
//! claimed, which is exactly what a read of one of its buffers has to fail on.

use crate::id::KernelId;
use crate::memory_management::FailureId;
use crate::server::{BufferBinding, ServerError};
use crate::stream::{FailureStore, StreamCapture};
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// A server whose device work runs inside an [`ExecuteScope`].
///
/// A server supplies the three things a scope cannot know — where its
/// multi-stream driver lives, what a failure means to a measurement in
/// flight, and whether the stream is recording a graph — and gets the scope
/// for it, which is the only way its launch and host-copy paths should touch
/// the failure bookkeeping.
pub trait WriteScoped: Sized {
    /// The multi-stream driver the server keeps.
    type Streams: FailureStore;

    /// The driver, split-borrowed from the server so the scope can claim on
    /// the way in and settle on the way out while `body` holds the rest.
    fn write_streams(&mut self) -> &mut Self::Streams;

    /// Told about every failure a scope settles, so a measurement in flight
    /// on `stream` is invalidated wherever the failure happened.
    ///
    /// A failed candidate that benchmarked at close to zero would otherwise
    /// win a tune. The stream is named because some backends keep a
    /// measurement per stream and some keep one per device; the scope knows
    /// which stream its work was for, and the server knows which of those it
    /// is. Defaults to doing nothing, for a server that measures nothing.
    #[allow(unused_variables)]
    fn on_failure(&mut self, stream: StreamId, error: &ServerError) {}

    /// The graph capture of the pooled stream `stream` folds onto, for a
    /// server that records captures. Defaults to `None`, for one that does
    /// not.
    ///
    /// The scope is the window's only informant. Work that fails or is
    /// skipped inside a recording window dooms it through this accessor —
    /// the recording is missing an operation and must not seal, and the
    /// replay contract has the caller write fresh inputs before each replay,
    /// clearing the very claim that would explain the hole. Work that exits
    /// clean hands the window its write set, so "recorded" and "in the
    /// graph" are the same event and cannot disagree. A server that returns
    /// its capture state gets all of that without wiring any of it.
    ///
    /// The stream is the one the work was issued on, which the server cannot
    /// work out for itself: it runs on its own thread, so the caller's
    /// current stream is not this one.
    #[allow(unused_variables)]
    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        None
    }

    /// An empty write set for the caller to fill with what the work it is
    /// about to run writes.
    ///
    /// Filled by the caller rather than inside the scope, which is what lets
    /// the body take the arguments the set was read from by value. A set left
    /// empty claims nothing, which is what a dry run wants.
    fn write_set(&mut self) -> Vec<BufferBinding> {
        self.write_streams().write_set()
    }
}

/// What a scope's work did.
#[derive(Debug)]
pub enum ScopedOutcome<R> {
    /// It ran, and its write set has a writer again.
    Executed(R),
    /// It did not run, because an input it needed carried a failure. Its
    /// write set carries that same failure now, so a read of one of those
    /// buffers names the original cause.
    Skipped,
    /// It ran and failed. Its write set carries the error.
    Failed(ServerError),
}

impl<R> ScopedOutcome<R> {
    /// The result, when the work ran and succeeded.
    ///
    /// # Errors
    ///
    /// The error it failed with. A skip answers with an error saying so — the
    /// failure its inputs carried is not the caller's to receive here; the
    /// claim on the write set is the report, and a read of one of those
    /// buffers names the root cause.
    pub fn into_result(self) -> Result<R, ServerError> {
        match self {
            ScopedOutcome::Executed(result) => Ok(result),
            ScopedOutcome::Failed(error) => Err(error),
            ScopedOutcome::Skipped => Err(ServerError::Skipped),
        }
    }
}

/// Whether the scope claimed its write set for work about to run, or pointed
/// it at the failure that stopped the work happening at all.
enum Opened {
    /// Entered: the write set carries a provisional failure until the exit
    /// replaces or releases it. `None` when the set was empty, which claims
    /// and mints nothing.
    Entered {
        provisional: Option<FailureId>,
        written: Vec<BufferBinding>,
    },
    /// Skipped: the write set already carries the failure its inputs did, and
    /// nothing else is owed.
    Skipped,
}

/// One scope around a unit of device work.
///
/// Built by [`over`](Self::over) for work that reads nothing it must trust, or
/// [`launching`](Self::launching) for a kernel, which is the only kind that
/// can be skipped.
///
/// Which of the two it is is settled by the constructor and cannot change
/// afterwards. Entering on a skip would be wrong rather than merely wasteful:
/// the provisional failure would be minted, overwritten by the propagated one,
/// and never pruned, because the exit that prunes it is on the path that does
/// not run. Nothing has to remember that, because a scope is one or the other
/// before it exists.
pub struct ExecuteScope<'a, S: WriteScoped> {
    server: &'a mut S,
    /// The stream this work is for, which is what a failure has to name.
    stream: StreamId,
    opened: Opened,
}

impl<'a, S: WriteScoped> ExecuteScope<'a, S> {
    /// A scope over work that writes `written` and reads nothing it has to
    /// trust — a host copy, a graph replay, a launch that never compiled.
    ///
    /// Such work cannot be skipped, so this always enters.
    pub fn over(server: &'a mut S, stream: StreamId, written: Vec<BufferBinding>) -> Self {
        let provisional = server.write_streams().enter_write(&written);
        Self {
            server,
            stream,
            opened: Opened::Entered {
                provisional,
                written,
            },
        }
    }

    /// A scope over a launch of `kernel` on `stream`, reading `reads` and
    /// writing `written`.
    ///
    /// Skips, rather than claiming, when an input carries a failure. A launch
    /// whose input cannot be trusted does not run: a buffer holding garbage
    /// can be read as a dynamic cube count or as gather indices, scattering
    /// into memory that carried no failure at all. Its outputs take the
    /// failure that stopped it, exactly as a failed launch's would, so a read
    /// downstream fails on the root cause.
    pub fn launching<'b>(
        server: &'a mut S,
        kernel: KernelId,
        stream: StreamId,
        reads: impl Iterator<Item = &'b BufferBinding>,
        written: Vec<BufferBinding>,
    ) -> Self {
        let Some(found) = server.write_streams().read_failure(reads) else {
            return Self::over(server, stream, written);
        };
        server.on_failure(stream, &found.error);
        // A skip inside a recording window dooms it: the recording is missing
        // this launch and must not seal. A no-op outside one.
        if let Some(capture) = server.capturing(stream) {
            capture.fail(found.error.clone());
        }
        server.write_streams().propagate(&found, kernel, written);
        Self {
            server,
            stream,
            opened: Opened::Skipped,
        }
    }

    /// Whether this scope skipped, so its body will not run.
    pub fn skipped(&self) -> bool {
        matches!(self.opened, Opened::Skipped)
    }

    /// Run `body` and settle the write set.
    ///
    /// A skipped scope never runs it. Otherwise the claim is released if the
    /// body succeeded and replaced with the real error if it did not, and
    /// either way a measurement in flight hears about a failure, and a
    /// recording window hears how the work ended. `body` may return early
    /// anywhere.
    pub fn execute<R>(
        self,
        body: impl FnOnce(&mut S) -> Result<R, ServerError>,
    ) -> ScopedOutcome<R> {
        let Opened::Entered {
            provisional,
            written,
        } = self.opened
        else {
            return ScopedOutcome::Skipped;
        };

        let result = body(self.server);
        // The recording window hears the exit before the claim settles, from
        // the one place that always knows how the work ended. A clean exit
        // hands it the write set — what the graph will write is what a scope
        // inside it wrote, recorded here so the two cannot disagree. A failed
        // exit dooms it: the recording is missing this work and must not
        // seal. Both are no-ops outside a window.
        if let Some(capture) = self.server.capturing(self.stream) {
            match result.as_ref() {
                Ok(_) => capture.record(written.iter().cloned()),
                Err(error) => capture.fail(error.clone()),
            }
        }
        self.server
            .write_streams()
            .exit_write(provisional, written, result.as_ref().err());
        match result {
            Ok(result) => ScopedOutcome::Executed(result),
            Err(error) => {
                self.server.on_failure(self.stream, &error);
                ScopedOutcome::Failed(error)
            }
        }
    }
}

/// Claim `written` for `error` without running anything: work that never
/// started leaves its destinations exactly as they were, so a read of one of
/// them has to fail on the error that stopped it.
pub fn failed_writing<S: WriteScoped>(
    server: &mut S,
    stream: StreamId,
    written: Vec<BufferBinding>,
    error: ServerError,
) {
    let _ = ExecuteScope::over(server, stream, written).execute(|_| Err::<(), _>(error));
}
