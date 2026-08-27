//! One scope around the writes of a unit of device work.
//!
//! Every backend used to keep the same bookkeeping by hand: gather what the
//! work writes, taint it on every failure path, release it on the success
//! path, and hope no early return forgot. The scope makes forgetting loud
//! instead of silent. Entering taints the write set with a provisional
//! failure — the real one does not exist yet — and leaving either releases it
//! or swaps the real error in. A path that never reaches the exit, a panic
//! mid-launch above all, leaves the write set tainted, so a read of one of
//! its buffers fails loudly instead of returning bytes nothing wrote.
//!
//! It is a closure and not a guard value because a guard enforces nothing
//! here: `#[must_use]` says nothing about a bound value on a path that
//! returns early, and a `Drop` implementation cannot reach the failure store
//! and must not assert during an unwind. A closure makes the early return
//! structurally impossible.

use crate::memory_management::FailureId;
use crate::server::{BufferBinding, ServerError};
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// The multi-stream driver a write scope stages, taints and settles through.
///
/// Implemented by the drivers that own the device's
/// [`ErrorGraph`](crate::memory_management::ErrorGraph) and its streams; the
/// real work happens on [`StreamPool::enter_write`] and
/// [`StreamPool::exit_write`], and this trait is how a scope reaches them
/// without knowing which driver the server keeps.
///
/// [`StreamPool::enter_write`]: super::StreamPool::enter_write
/// [`StreamPool::exit_write`]: super::StreamPool::exit_write
pub trait WriteStreams {
    /// The vector the scope stages its write set in, pooled on the driver so
    /// a launch allocates little for it. [`exit`](Self::exit) hands it back.
    fn stage(&mut self) -> Vec<BufferBinding>;

    /// Taint every buffer in `written` with a provisional failure, minted
    /// because the real one does not exist yet. `None` when the set is empty:
    /// a dry run enters and leaves having claimed nothing.
    fn enter(&mut self, written: &[BufferBinding]) -> Option<FailureId>;

    /// Settle the scope: release the provisional failure when `error` is
    /// `None`, swap the real error in for it and queue it on `stream_id`
    /// otherwise — and return the staged vector to the pool either way.
    fn exit(
        &mut self,
        provisional: Option<FailureId>,
        written: Vec<BufferBinding>,
        stream_id: StreamId,
        error: Option<&ServerError>,
    );
}

/// A server whose device work runs inside a write scope.
///
/// A server supplies the one thing the scope cannot know — where its
/// multi-stream driver lives — and gets [`while_writing`] for it, which is
/// the only way its launch and host-copy paths should touch the taint
/// bookkeeping.
///
/// [`while_writing`]: Self::while_writing
pub trait WriteScoped: Sized {
    /// The multi-stream driver the server keeps.
    type Streams: WriteStreams;

    /// The driver, split-borrowed from the server so the scope can taint on
    /// the way in and settle on the way out while `body` holds the rest.
    fn write_streams(&mut self) -> &mut Self::Streams;

    /// Run `body` inside a scope over what the work writes.
    ///
    /// `writes` names the write set from `payload` — the arguments the body
    /// consumes, handed through by value because the set borrows from them
    /// and the body needs them back. It stages nothing for work that writes
    /// nothing, a dry run above all. `body` does the device work and may
    /// return early anywhere; it also receives the staged set, for the
    /// backends that record it into a capture.
    ///
    /// On entry every staged buffer is tainted with a provisional failure. On
    /// exit the taint is released if `body` succeeded, and replaced with the
    /// real error — queued on `stream_id` for its next flush to report — if
    /// it did not. A body that panics never reaches the exit, and the
    /// provisional taint is exactly what it leaves behind.
    fn while_writing<A, R>(
        &mut self,
        stream_id: StreamId,
        payload: A,
        writes: impl FnOnce(&A, &mut Vec<BufferBinding>),
        body: impl FnOnce(&mut Self, A, &[BufferBinding]) -> Result<R, ServerError>,
    ) -> Result<R, ServerError> {
        let mut written = self.write_streams().stage();
        writes(&payload, &mut written);
        let provisional = self.write_streams().enter(&written);
        let result = body(self, payload, &written);
        self.write_streams()
            .exit(provisional, written, stream_id, result.as_ref().err());
        result
    }
}
