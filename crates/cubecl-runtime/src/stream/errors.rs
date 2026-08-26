//! The per-stream error queue every backend records failures in, and the
//! attribution rule that decides which logical stream surfaces each one.

use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

use crate::memory_management::ManagedMemoryId;
use crate::server::ServerError;

/// The errors queued on one pooled backend stream, each remembering the logical
/// stream it belongs to and the buffers it left unwritten.
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
/// # What a read asks
///
/// Attribution answers *who* surfaces an error. It does not answer the question
/// a read has to ask before copying bytes out: *were these bytes ever written?*
/// A stream id cannot answer it. [`Handle::stream`](crate::server::Handle) names
/// where a buffer was **created**, not who last wrote it, and nothing re-tags it
/// — so a buffer allocated on A, written by a launch that failed on B, and read
/// back on A names only A, and A has nothing queued.
///
/// So the failures that leave a buffer unwritten name the buffers themselves,
/// with [`push_unwritten`](Self::push_unwritten), and a read asks every slot
/// [`peek_unwritten`](Self::peek_unwritten) for the buffers it is about to copy.
/// That holds whichever stream allocated, wrote or read them.
///
/// # What is bounded, and what is not
///
/// Attribution is best-effort in one direction only. Owned entries are capped
/// at [`MAX_OWNED`]: past that the oldest fall back to shared, so a stream that
/// queues an error and never comes back cannot pin memory for the life of the
/// process. Shared entries are not capped here — the next flush of any stream
/// drains them, which is the bound in practice.
///
/// The fallback costs exactly one thing, which is safe. A reclaimed entry is
/// surfaced by whichever stream flushes next rather than by the one that caused
/// it, so the report no longer names the producer. It keeps its unwritten
/// buffers, so a read that would have failed on it still fails; and it makes
/// [`any`](Self::any) true for every stream on the slot until someone flushes,
/// so a neighbour is briefly reported unhealthy.
///
/// [`stream_index`]: super::stream_index
#[derive(Debug, Default)]
pub struct StreamErrors {
    entries: Vec<Entry>,
}

/// One queued failure: who surfaces it, and what it left unwritten.
#[derive(Debug)]
struct Entry {
    /// The logical stream that caused it, or `None` for an error the slot
    /// cannot attribute — which any stream's flush surfaces.
    owner: Option<StreamId>,
    error: ServerError,
    /// The buffers the failed work never wrote, empty when the failure left
    /// none behind.
    unwritten: Vec<ManagedMemoryId>,
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
        self.entries.push(Entry {
            owner: Some(owner),
            error,
            unwritten: Vec::new(),
        });
        self.reclaim_orphans();
    }

    /// Queue an error caused by `owner` that left `unwritten` never written.
    ///
    /// For the failures that skip work a buffer was waiting on: a launch that
    /// never reached the device, a host copy that never started. A read of any
    /// of those buffers would hand back whatever was in memory before, so it
    /// fails on this error instead — see [`peek_unwritten`](Self::peek_unwritten).
    ///
    /// A launch names every buffer it was given, inputs included: a server sees
    /// resources, not which of them the kernel would have written. Erring wide
    /// costs a read of an untouched input the error its own launch already
    /// queued; erring narrow would hand back an output nothing wrote.
    pub fn push_unwritten(
        &mut self,
        owner: StreamId,
        error: ServerError,
        unwritten: impl IntoIterator<Item = ManagedMemoryId>,
    ) {
        self.entries.push(Entry {
            owner: Some(owner),
            error,
            unwritten: unwritten.into_iter().collect(),
        });
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
        self.entries.push(Entry {
            owner: None,
            error,
            unwritten: Vec::new(),
        });
    }

    /// [`push_shared`](Self::push_shared) for a batch of errors.
    pub fn extend_shared(&mut self, errors: impl IntoIterator<Item = ServerError>) {
        self.entries.extend(errors.into_iter().map(|error| Entry {
            owner: None,
            error,
            unwritten: Vec::new(),
        }));
    }

    /// Whether `owner` has anything to surface: its own errors plus the shared
    /// ones.
    pub fn any(&self, owner: StreamId) -> bool {
        self.entries
            .iter()
            .any(|entry| surfaced_by(entry.owner, owner))
    }

    /// The queued errors that left one of `buffers` unwritten, other than
    /// `reader`'s own, left in the queue.
    ///
    /// This is the question a read asks every pooled stream before it copies
    /// anything: did the work that was supposed to write these bytes fail? An
    /// entry answers it by naming the buffer, not the stream, so the answer
    /// holds however the buffer travelled between streams.
    ///
    /// `reader`'s own entries are left out because the read flushes `reader` on
    /// its way through, which takes them and reports them once. The entries
    /// this returns stay put, so the stream that caused each one still surfaces
    /// it itself.
    pub fn peek_unwritten(
        &self,
        buffers: &[ManagedMemoryId],
        reader: StreamId,
    ) -> Vec<ServerError> {
        self.entries
            .iter()
            .filter(|entry| entry.owner != Some(reader))
            .filter(|entry| entry.unwritten.iter().any(|id| buffers.contains(id)))
            .map(|entry| entry.error.clone())
            .collect()
    }

    /// Take the errors `owner` surfaces, leaving the other streams' behind.
    pub fn take(&mut self, owner: StreamId) -> Vec<ServerError> {
        let (taken, kept): (Vec<_>, Vec<_>) = core::mem::take(&mut self.entries)
            .into_iter()
            .partition(|entry| surfaced_by(entry.owner, owner));

        self.entries = kept;
        taken.into_iter().map(|entry| entry.error).collect()
    }

    /// Re-tag the oldest owned entries as shared once more than [`MAX_OWNED`]
    /// are waiting, so entries whose owner is gone stop accumulating.
    ///
    /// Only the attribution is dropped. The buffers the entry left unwritten
    /// stay on it, so a read that those bytes concern still fails on it.
    fn reclaim_orphans(&mut self) {
        let owned = self.entries.iter().filter(|e| e.owner.is_some()).count();
        if owned <= MAX_OWNED {
            return;
        }
        let mut excess = owned - MAX_OWNED;

        for entry in self.entries.iter_mut() {
            if entry.owner.is_some() {
                entry.owner = None;
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

    /// What lets a read consult the work that wrote its buffers without
    /// stealing that work's errors.
    ///
    /// A reader must see that a producer's launch failed — the bytes it is
    /// about to copy were never written — but taking the error would leave the
    /// stream that actually failed reporting success on its own flush.
    #[test]
    fn an_unwritten_buffer_is_read_without_taking_its_error() {
        let producer = StreamId { value: 1 };
        let reader = StreamId { value: 2 };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(producer, error("launch"), [buffer]);
        errors.push_shared(error("submission"));

        // The reader sees what the failed launch left unwritten, but not the
        // shared entry, which speaks for no particular buffer.
        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], reader)),
            ["launch"]
        );
        assert_eq!(reasons(errors.take(reader)), ["submission"]);
        assert_eq!(
            reasons(errors.take(producer)),
            ["launch"],
            "the producer still surfaces its own error"
        );
    }

    /// The case a stream id cannot answer: the buffer was allocated on the
    /// stream now reading it, and written by a launch that failed elsewhere.
    ///
    /// `Handle::stream` names where a buffer was created, and nothing re-tags
    /// it, so both the allocation and the read name the reader — which has
    /// nothing queued. Only the buffer itself connects the read to the launch
    /// that never ran.
    #[test]
    fn a_buffer_the_reader_allocated_still_answers_for_another_streams_launch() {
        let reader = StreamId { value: 1 };
        let producer = StreamId { value: 2 };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(producer, error("launch"), [buffer]);

        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], reader)),
            ["launch"]
        );
    }

    /// A read only fails on the buffers the failure actually concerns.
    ///
    /// A pending error on some stream is not a reason to refuse every read: the
    /// stream that queued it may never flush again, which would leave every
    /// later read of an unrelated buffer failing on it forever.
    #[test]
    fn buffers_a_failure_never_touched_are_read_normally() {
        let producer = StreamId { value: 1 };
        let reader = StreamId { value: 2 };
        let unwritten = ManagedMemoryId { value: 10 };
        let untouched = ManagedMemoryId { value: 11 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(producer, error("launch"), [unwritten]);
        // A failure that left no buffer behind concerns no read at all.
        errors.push(producer, error("profile"));

        assert!(errors.peek_unwritten(&[untouched], reader).is_empty());
        assert_eq!(
            reasons(errors.peek_unwritten(&[unwritten], reader)),
            ["launch"]
        );
    }

    /// The reader's own failures are left to the flush the read already does,
    /// so they are reported once rather than twice.
    #[test]
    fn the_readers_own_errors_are_left_to_its_flush() {
        let reader = StreamId { value: 1 };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(reader, error("launch"), [buffer]);

        assert!(errors.peek_unwritten(&[buffer], reader).is_empty());
        assert_eq!(reasons(errors.take(reader)), ["launch"]);
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
            errors.entries.iter().filter(|e| e.owner.is_some()).count(),
            MAX_OWNED,
            "the queue stops holding entries for streams that are long gone"
        );
        // Nothing was dropped on the way: the reclaimed entries are shared, so
        // the next flush of any stream still surfaces them.
        let bystander = StreamId { value: u64::MAX };
        assert_eq!(errors.take(bystander).len(), MAX_OWNED * 3);
    }

    /// Losing the attribution must not lose the buffers: a read of what a
    /// reclaimed entry left unwritten still fails, whichever stream asks.
    ///
    /// A reclaimed entry lives on the stream that caused it, and the reader may
    /// be on any other pooled slot — so leaving the read to the flush that
    /// eventually drains it would hand back bytes nothing wrote in the meantime.
    #[test]
    fn a_reclaimed_entry_keeps_the_buffers_it_left_unwritten() {
        let reader = StreamId { value: u64::MAX };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(StreamId { value: 0 }, error("launch"), [buffer]);
        // Push past the cap so the entry above is re-tagged as shared.
        for value in 1..(MAX_OWNED as u64 * 2) {
            errors.push(StreamId { value }, error("other"));
        }

        assert_eq!(
            errors.entries[0].owner, None,
            "the oldest entry was reclaimed"
        );
        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], reader)),
            ["launch"]
        );
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
