//! The per-stream error queue every backend records failures in, and the
//! attribution rule that decides which logical stream surfaces each one.

use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

use crate::server::ServerError;

/// The errors queued on one pooled backend stream, each remembering the logical
/// stream it belongs to.
///
/// Several logical streams share a pooled stream: [`stream_index`] folds the id
/// space onto `max_streams` slots, so thread A and thread B can land on the same
/// backend stream. Errors are lazy — a failed launch is queued and surfaces at
/// the next flush, sync or profile end — so a single untagged queue per slot
/// lets B drain A's error: B panics on a kernel it never launched while A reads
/// back a zeroed buffer as if its launch had succeeded.
///
/// Tagging fixes the attribution: [`push`](Self::push) records the stream that
/// caused the error and only that stream ever takes it. Errors a slot cannot
/// attribute — a failed submission, a mapping failure, a validation message
/// from the driver — go in with [`push_shared`](Self::push_shared) and surface
/// on whichever stream flushes next.
///
/// # What is bounded, and what is not
///
/// Attribution is best-effort in one direction only. Owned entries are capped
/// at [`MAX_OWNED`]: past that the oldest fall back to shared, so a stream that
/// queues an error and never comes back cannot pin memory for the life of the
/// process. Shared entries are not capped here — the next flush of any stream
/// drains them, which is the bound in practice.
///
/// The fallback costs two things, both safe. A reclaimed entry stops answering
/// [`peek_owned`](Self::peek_owned), so the read path no longer names the
/// producer that failed — but the reader's own flush takes shared entries, so
/// the read still fails rather than handing back bytes nothing wrote. And a
/// reclaimed entry makes [`any`](Self::any) true for every stream on the slot
/// until someone flushes, so a neighbour is briefly reported unhealthy.
///
/// [`stream_index`]: super::stream_index
#[derive(Debug, Default)]
pub struct StreamErrors {
    entries: Vec<(Option<StreamId>, ServerError)>,
}

/// How many entries one queue keeps waiting on the streams that own them.
///
/// Logical stream ids are unbounded and short-lived — under
/// [`StreamPolicy::PerTask`](cubecl_environment::stream::StreamPolicy) every
/// task gets its own — so a stream that queues an error and is dropped before
/// it flushes leaves an entry nothing will ever match again. Past this many
/// waiting entries the oldest are re-tagged as shared, which the next flush of
/// any stream reclaims: attribution is lost, but no error is dropped and the
/// owned half of the queue stops growing.
pub const MAX_OWNED: usize = 32;

impl StreamErrors {
    /// Queue an error caused by `owner`, for `owner` alone to surface.
    pub fn push(&mut self, owner: StreamId, error: ServerError) {
        self.entries.push((Some(owner), error));
        self.reclaim_orphans();
    }

    /// Queue an error raised by synchronizing the backend stream on `caller`'s
    /// behalf.
    ///
    /// A [`ServerError::ServerUnhealthy`] is the queue the flush on the way in
    /// already took from `caller`, so it goes back to `caller`. Anything else
    /// comes from the device — the synchronize itself failed, leaving a context
    /// every logical stream sharing this backend stream keeps hitting — so it
    /// goes in unattributed rather than pinned on the stream that happened to
    /// ask.
    pub fn push_sync_failure(&mut self, caller: StreamId, error: ServerError) {
        match error {
            ServerError::ServerUnhealthy { .. } => self.push(caller, error),
            error => self.push_shared(error),
        }
    }

    /// Queue an error the pooled stream cannot attribute to one of the logical
    /// streams sharing it, so the next flush of any of them surfaces it.
    pub fn push_shared(&mut self, error: ServerError) {
        self.entries.push((None, error));
    }

    /// [`push_shared`](Self::push_shared) for a batch of errors.
    pub fn extend_shared(&mut self, errors: impl IntoIterator<Item = ServerError>) {
        self.entries
            .extend(errors.into_iter().map(|error| (None, error)));
    }

    /// Whether `owner` has anything to surface: its own errors plus the shared
    /// ones.
    pub fn any(&self, owner: StreamId) -> bool {
        self.entries
            .iter()
            .any(|(entry, _)| surfaced_by(*entry, owner))
    }

    /// The errors `owner` alone caused, left in the queue — its own, without
    /// the shared ones any stream may surface.
    ///
    /// This is the question one stream asks about *another*: did the work that
    /// wrote this buffer fail? Only entries that stream owns answer it, and the
    /// entry stays put so the stream that caused it still surfaces it itself.
    pub fn peek_owned(&self, owner: StreamId) -> Vec<ServerError> {
        self.entries
            .iter()
            .filter(|(entry, _)| *entry == Some(owner))
            .map(|(_, error)| error.clone())
            .collect()
    }

    /// Take the errors `owner` surfaces, leaving the other streams' behind.
    pub fn take(&mut self, owner: StreamId) -> Vec<ServerError> {
        let (taken, kept) = core::mem::take(&mut self.entries)
            .into_iter()
            .partition(|(entry, _)| surfaced_by(*entry, owner));

        self.entries = kept;
        taken.into_iter().map(|(_, error)| error).collect()
    }

    /// Re-tag the oldest owned entries as shared once more than [`MAX_OWNED`]
    /// are waiting, so entries whose owner is gone stop accumulating.
    fn reclaim_orphans(&mut self) {
        let owned = self.entries.iter().filter(|(o, _)| o.is_some()).count();
        if owned <= MAX_OWNED {
            return;
        }
        let mut excess = owned - MAX_OWNED;

        for (owner, _) in self.entries.iter_mut() {
            if owner.is_some() {
                *owner = None;
                excess -= 1;
                if excess == 0 {
                    return;
                }
            }
        }
    }
}

/// Whether `owner` surfaces an entry queued for `entry`: its own, plus the
/// shared entries no stream owns.
fn surfaced_by(entry: Option<StreamId>, owner: StreamId) -> bool {
    match entry {
        Some(entry) => entry == owner,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// The attribution the whole type exists for: a stream surfaces the errors
    /// its own work caused and nothing else.
    ///
    /// Draining a neighbour's error fails it on a kernel it never launched, and
    /// leaves the stream that did launch it reading back a buffer nothing wrote
    /// as if all was well.
    #[test]
    fn a_stream_takes_its_own_errors_only() {
        let first = StreamId { value: 1 };
        let second = StreamId { value: 2 };

        let mut errors = StreamErrors::default();
        errors.push(first, error("first"));
        errors.push(second, error("second"));

        assert_eq!(reasons(errors.take(first)), ["first"]);
        assert!(errors.any(second), "the other stream keeps its error");
        assert_eq!(reasons(errors.take(second)), ["second"]);
        assert!(!errors.any(first) && !errors.any(second));
    }

    /// An error the slot cannot attribute is still reported, exactly once.
    ///
    /// A driver validation message or a failed submission names no logical
    /// stream. Holding it back until its owner asks would drop it, since no
    /// owner is ever coming.
    #[test]
    fn shared_errors_surface_on_the_next_flush() {
        let stream = StreamId { value: 1 };

        let mut errors = StreamErrors::default();
        errors.push_shared(error("submission"));

        assert!(
            errors.any(stream),
            "a stream that caused nothing still sees it"
        );
        assert_eq!(reasons(errors.take(stream)), ["submission"]);
        assert!(!errors.any(stream), "a shared error is taken once");
    }

    /// What lets a read consult the streams that wrote its buffers without
    /// stealing their errors.
    ///
    /// A reader must see that a producer's launch failed — the bytes it is
    /// about to copy were never written — but taking the error would leave the
    /// stream that actually failed reporting success on its own flush.
    #[test]
    fn another_streams_errors_are_read_without_taking_them() {
        let producer = StreamId { value: 1 };
        let reader = StreamId { value: 2 };

        let mut errors = StreamErrors::default();
        errors.push(producer, error("launch"));
        errors.push_shared(error("submission"));

        // The reader sees what the producer's failed work left behind, but not
        // the shared entry, which speaks for no particular stream's work.
        assert_eq!(reasons(errors.peek_owned(producer)), ["launch"]);
        assert_eq!(reasons(errors.take(reader)), ["submission"]);
        assert_eq!(
            reasons(errors.take(producer)),
            ["launch"],
            "the producer still surfaces its own error"
        );
    }

    /// A faulted device context is everyone's problem; a queue handed back to
    /// the caller is only the caller's.
    ///
    /// Pinning a synchronize failure on the stream that happened to ask would
    /// report every other stream on the faulted context as healthy.
    #[test]
    fn a_sync_failure_is_shared_unless_it_is_the_callers_own_queue() {
        let caller = StreamId { value: 1 };
        let other = StreamId { value: 2 };

        let mut errors = StreamErrors::default();
        // A device-level synchronize failure faults the context for everyone.
        errors.push_sync_failure(caller, error("cuStreamSynchronize failed"));
        assert_eq!(
            reasons(errors.take(other)),
            ["cuStreamSynchronize failed"],
            "a stream that did not ask for the sync still hits the faulted context"
        );

        // The errors a flush already took from the caller go back to it alone.
        errors.push_sync_failure(
            caller,
            ServerError::ServerUnhealthy {
                errors: alloc::vec![error("launch")],
                backtrace: Default::default(),
            },
        );
        assert!(!errors.any(other));
        assert!(errors.any(caller));
    }

    /// The queue stays bounded even though logical stream ids are not.
    ///
    /// Under `StreamPolicy::PerTask` a cancelled task queues an error and never
    /// flushes again, so entries waiting on owners that will never ask would
    /// pin a backtrace each for the life of the process. Reclaiming loses the
    /// attribution, never the error.
    #[test]
    fn orphaned_errors_are_reclaimed_rather_than_accumulated() {
        let mut errors = StreamErrors::default();

        // Every push comes from a stream that never flushes again, as a task
        // that is cancelled after a failed launch does.
        for value in 0..(MAX_OWNED as u64 * 4) {
            errors.push(StreamId { value }, error("launch"));
        }

        assert_eq!(
            errors.entries.iter().filter(|(o, _)| o.is_some()).count(),
            MAX_OWNED,
            "the queue stops holding entries for streams that are long gone"
        );
        // Nothing was dropped on the way: the reclaimed entries are shared, so
        // the next flush of any stream still surfaces them.
        let bystander = StreamId { value: u64::MAX };
        assert_eq!(errors.take(bystander).len(), MAX_OWNED * 3);
    }

    /// One flush drains both halves of what a stream is owed, and nothing else.
    ///
    /// A caller gets one report per flush, so an unattributed error held back
    /// until some later call would be a second failure for the same work; an
    /// owned error swept up with it would fail this stream for a launch it
    /// never made.
    #[test]
    fn a_flush_takes_the_shared_errors_along_with_its_own() {
        let flushing = StreamId { value: 1 };
        let other = StreamId { value: 2 };

        let mut errors = StreamErrors::default();
        errors.push(flushing, error("launch"));
        errors.push_shared(error("submission"));
        errors.push(other, error("other launch"));

        assert_eq!(reasons(errors.take(flushing)), ["launch", "submission"]);
        assert_eq!(reasons(errors.take(other)), ["other launch"]);
    }

    fn error(reason: &str) -> ServerError {
        ServerError::Generic {
            reason: reason.to_string(),
            backtrace: Default::default(),
        }
    }

    fn reasons(errors: Vec<ServerError>) -> Vec<alloc::string::String> {
        errors
            .into_iter()
            .map(|error| match error {
                ServerError::Generic { reason, .. } => reason,
                other => panic!("unexpected error: {other}"),
            })
            .collect()
    }
}
