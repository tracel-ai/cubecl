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
    /// `None`, swap the real error in for it and log it otherwise — and
    /// return the staged vector to the pool either way.
    fn exit(
        &mut self,
        provisional: Option<FailureId>,
        written: Vec<BufferBinding>,
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

    /// An empty write set, pooled on the driver, for the caller to fill with
    /// what the work it is about to run writes. Filling it here rather than
    /// inside the scope is what lets the body take the arguments the set was
    /// read from by value. A set left empty claims nothing, which is what a
    /// dry run wants.
    fn write_set(&mut self) -> Vec<BufferBinding> {
        self.write_streams().stage()
    }

    /// Run `body` inside a scope over `written`, the set from
    /// [`write_set`](Self::write_set).
    ///
    /// On entry every buffer in it is tainted with a provisional failure. On
    /// exit the taint is released if `body` succeeded, and replaced with the
    /// real error — logged there, since a read of the claimed buffers is what
    /// reports it — if it did not. `body` may return early anywhere; it also
    /// receives the set, for the backends that record it into a capture. A
    /// body that panics never reaches the exit, and the provisional taint is
    /// exactly what it leaves behind.
    fn while_writing<R>(
        &mut self,
        written: Vec<BufferBinding>,
        body: impl FnOnce(&mut Self, &[BufferBinding]) -> Result<R, ServerError>,
    ) -> Result<R, ServerError> {
        let provisional = self.write_streams().enter(&written);
        let result = body(self, &written);
        self.write_streams()
            .exit(provisional, written, result.as_ref().err());
        result
    }

    /// Claim `written` for `error` without running anything: work that never
    /// started leaves its destinations exactly as they were, so a read of one
    /// of them has to fail on the error that stopped it. A scope whose body
    /// is the failure itself.
    fn failed_writing(&mut self, written: Vec<BufferBinding>, error: ServerError) {
        let _ = self.while_writing(written, |_, _| Err::<(), _>(error));
    }
}
