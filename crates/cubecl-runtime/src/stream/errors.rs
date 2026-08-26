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
/// The two jobs come apart entirely for a failure its caller already has in
/// hand: [`push_returned`](Self::push_returned) answers reads and no flush ever
/// reports it, so a caller that was told synchronously is not told again.
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

/// One queued failure: whose flush surfaces it, and what it left unwritten.
#[derive(Debug)]
struct Entry {
    surfaced: Surfaced,
    error: ServerError,
    /// The buffers the failed work never wrote, empty when the failure left
    /// none behind.
    unwritten: Vec<ManagedMemoryId>,
}

/// Which flush reports a queued error.
///
/// Reporting and answering a read are separate jobs, and the third variant is
/// the one that only does the second: an error already handed to its caller
/// must not be handed to anyone a second time, but the buffers it left
/// unwritten still concern every stream that has not heard about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Surfaced {
    /// This logical stream's next flush, and no other's.
    Owner(StreamId),
    /// Whichever stream flushes next — the slot could not attribute it.
    AnyStream,
    /// None of them. The error was returned to its caller as it happened, so
    /// the entry stays only so that a read of what it left unwritten fails.
    Returned,
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

/// How many already-returned entries one queue keeps to answer reads.
///
/// Nothing drains an already-returned entry — that is the point of it — so the
/// cap is its only bound. Past it the oldest are dropped, which costs exactly
/// what a flush costs an ordinary entry: the buffers it named stop failing
/// reads. A failure that returns to its caller and still leaves a buffer behind
/// is rare enough (a capture that never sealed) that the newest few are the
/// ones that matter.
pub const MAX_RETURNED: usize = 32;

impl StreamErrors {
    /// Queue an error caused by `owner`, for `owner` alone to surface.
    pub fn push(&mut self, owner: StreamId, error: ServerError) {
        self.entries.push(Entry {
            surfaced: Surfaced::Owner(owner),
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
    /// resources, not which of them the kernel would have written. Erring narrow
    /// would hand back an output nothing wrote, silently; erring wide fails a
    /// read that would have been fine, loudly — so it errs wide.
    ///
    /// That cost is not confined to the stream that failed. An input another
    /// logical stream filled and reads back fails on this error too, from the
    /// moment it is queued until the stream that owns it flushes — under
    /// [`StreamPolicy::PerTask`](cubecl_environment::stream::StreamPolicy)
    /// possibly never, leaving [`MAX_OWNED`] reclaim plus somebody else's flush
    /// to clear it. Narrowing to the buffers a kernel actually writes needs
    /// per-binding visibility, which the launch path does not carry today.
    pub fn push_unwritten(
        &mut self,
        owner: StreamId,
        error: ServerError,
        unwritten: impl IntoIterator<Item = ManagedMemoryId>,
    ) {
        self.entries.push(Entry {
            surfaced: Surfaced::Owner(owner),
            error,
            unwritten: unwritten.into_iter().collect(),
        });
        self.reclaim_orphans();
    }

    /// Queue an error that has already been returned to its caller, so that a
    /// read of `unwritten` still fails on it and no flush reports it twice.
    ///
    /// For the failures a caller learns about synchronously and other streams
    /// do not: a graph capture that never sealed returns its error to whoever
    /// called `end_capture`, but the launches it recorded never ran, and the
    /// buffers they were given are read by streams that heard nothing. Queuing
    /// it the ordinary way would report it a second time on the caller's next
    /// flush — and on an abandoned window, to a stream that never asked.
    ///
    /// Bounded by [`MAX_RETURNED`] rather than by a flush, since no flush takes
    /// these, and ended by [`written`](Self::written) when the buffers it names
    /// are given to work that runs.
    pub fn push_returned(
        &mut self,
        error: ServerError,
        unwritten: impl IntoIterator<Item = ManagedMemoryId>,
    ) {
        let unwritten: Vec<_> = unwritten.into_iter().collect();
        if unwritten.is_empty() {
            // Nobody would report it and it answers for nothing: the caller
            // already holds the only copy that matters.
            return;
        }

        self.entries.push(Entry {
            surfaced: Surfaced::Returned,
            error,
            unwritten,
        });
        self.reclaim_returned();
    }

    /// Drop the returned entries' claim on `buffers`: work that writes them has
    /// been enqueued, so a read of one is no longer reading bytes nothing wrote.
    ///
    /// Already-returned entries are the only ones this concerns, because they
    /// are the only ones nothing else ends. An entry a flush still has to
    /// report keeps its claim until that flush takes it, which is its bound;
    /// stripping it here would shorten a window the flush already closes.
    ///
    /// A launch names every buffer it was given, inputs included — the same
    /// blindness [`push_unwritten`](Self::push_unwritten) errs wide against,
    /// read the other way round. An input that a failed capture named, and that
    /// a later launch only reads, stops failing reads from here rather than
    /// from its own next writer.
    pub fn written(&mut self, buffers: impl IntoIterator<Item = ManagedMemoryId>) {
        if self.returned() == 0 {
            // The common case, and worth the check: it costs the caller nothing
            // to offer the buffers of every launch when there is no capture
            // failure outstanding to clear.
            return;
        }

        for id in buffers {
            for entry in self.entries.iter_mut() {
                if entry.surfaced == Surfaced::Returned {
                    entry.unwritten.retain(|unwritten| *unwritten != id);
                }
            }
        }

        self.entries
            .retain(|entry| entry.surfaced != Surfaced::Returned || !entry.unwritten.is_empty());
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
            surfaced: Surfaced::AnyStream,
            error,
            unwritten: Vec::new(),
        });
    }

    /// [`push_shared`](Self::push_shared) for a batch of errors.
    pub fn extend_shared(&mut self, errors: impl IntoIterator<Item = ServerError>) {
        self.entries.extend(errors.into_iter().map(|error| Entry {
            surfaced: Surfaced::AnyStream,
            error,
            unwritten: Vec::new(),
        }));
    }

    /// Whether `owner` has anything to surface: its own errors plus the shared
    /// ones.
    pub fn any(&self, owner: StreamId) -> bool {
        self.entries
            .iter()
            .any(|entry| surfaced_by(entry.surfaced, owner))
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
            .filter(|entry| entry.surfaced != Surfaced::Owner(reader))
            .filter(|entry| entry.unwritten.iter().any(|id| buffers.contains(id)))
            .map(|entry| entry.error.clone())
            .collect()
    }

    /// Take the errors `owner` surfaces, leaving the other streams' behind.
    pub fn take(&mut self, owner: StreamId) -> Vec<ServerError> {
        let (taken, kept): (Vec<_>, Vec<_>) = core::mem::take(&mut self.entries)
            .into_iter()
            .partition(|entry| surfaced_by(entry.surfaced, owner));

        self.entries = kept;
        taken.into_iter().map(|entry| entry.error).collect()
    }

    /// Re-tag the oldest owned entries as shared once more than [`MAX_OWNED`]
    /// are waiting, so entries whose owner is gone stop accumulating.
    ///
    /// Only the attribution is dropped. The buffers the entry left unwritten
    /// stay on it, so a read that those bytes concern still fails on it.
    fn reclaim_orphans(&mut self) {
        let owned = self.owned();
        if owned <= MAX_OWNED {
            return;
        }
        let mut excess = owned - MAX_OWNED;

        for entry in self.entries.iter_mut() {
            if matches!(entry.surfaced, Surfaced::Owner(_)) {
                entry.surfaced = Surfaced::AnyStream;
                excess -= 1;
                if excess == 0 {
                    return;
                }
            }
        }
    }

    /// Drop the oldest already-returned entries once more than [`MAX_RETURNED`]
    /// are held, since no flush drains them.
    fn reclaim_returned(&mut self) {
        let returned = self.returned();
        if returned <= MAX_RETURNED {
            return;
        }
        let mut excess = returned - MAX_RETURNED;

        self.entries.retain(|entry| {
            if excess > 0 && entry.surfaced == Surfaced::Returned {
                excess -= 1;
                return false;
            }
            true
        });
    }

    /// How many entries are waiting on the streams that own them.
    fn owned(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.surfaced, Surfaced::Owner(_)))
            .count()
    }

    /// How many entries are held only to answer reads.
    fn returned(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.surfaced == Surfaced::Returned)
            .count()
    }
}

/// A backend stream that queues its failures in a [`StreamErrors`].
///
/// Every backend keeps one queue per pooled stream, some of them behind a lock,
/// and the multi-stream drivers ask it the same two questions on every read and
/// every resolve. Answering them here means a backend supplies only where its
/// queue lives, never what the drivers do with it.
pub trait StreamErrorSink {
    /// The queue this stream records its failures in.
    fn errors(&self) -> impl core::ops::Deref<Target = StreamErrors> + '_;

    /// Whether the stream can accept new work from `owner`.
    ///
    /// Broken for the streams whose errors are still queued on it, not for
    /// every stream sharing it — a neighbour that launched nothing is healthy
    /// however badly its slot-mate is doing.
    fn healthy(&self, owner: StreamId) -> bool {
        !self.errors().any(owner)
    }

    /// The queued errors that left one of `buffers` unwritten, other than
    /// `reader`'s own — see [`StreamErrors::peek_unwritten`].
    ///
    /// Answers what a read needs to know before it copies anything: did the
    /// work that was supposed to write these bytes actually run?
    fn unwritten(&self, buffers: &[ManagedMemoryId], reader: StreamId) -> Vec<ServerError> {
        self.errors().peek_unwritten(buffers, reader)
    }
}

/// Whether `owner`'s flush reports an entry: its own, plus the shared entries
/// no stream owns. Never an entry its caller already has.
fn surfaced_by(surfaced: Surfaced, owner: StreamId) -> bool {
    match surfaced {
        Surfaced::Owner(entry) => entry == owner,
        Surfaced::AnyStream => true,
        Surfaced::Returned => false,
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

    /// A failure its caller already has must not be reported to anyone a second
    /// time, and must still stop a read of what it left unwritten.
    ///
    /// A capture that never seals returns its error from `end_capture`; the
    /// launches it recorded never ran. Queuing that the ordinary way makes the
    /// caller's next flush fail on an error it was handed a moment earlier —
    /// which reads as a stream stuck in a bad state — while dropping it lets a
    /// neighbour read the buffers those launches never wrote.
    #[test]
    fn an_error_its_caller_already_has_is_not_surfaced_again() {
        let caller = StreamId { value: 1 };
        let neighbour = StreamId { value: 2 };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_returned(error("capture"), [buffer]);

        assert!(
            !errors.any(caller) && !errors.any(neighbour),
            "nobody has to report an error that was already returned"
        );
        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], neighbour)),
            ["capture"],
            "the buffers it left behind still concern a stream that never heard about it"
        );
        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], caller)),
            ["capture"],
            "including the caller: it holds the error, not the knowledge that this buffer is stale"
        );
        assert!(
            errors.take(caller).is_empty() && errors.take(neighbour).is_empty(),
            "a flush leaves it in place rather than reporting it"
        );
    }

    /// A returned entry lasts until something writes the buffer, and no longer.
    ///
    /// Nothing drains it — that is what makes it useful to a stream that never
    /// heard about the failure, and what would make it a permanent poison. The
    /// stream that owns a capture recovers by relaunching into the same
    /// buffers, and every read after that is reading what the relaunch wrote.
    #[test]
    fn a_returned_entry_ends_when_the_buffer_is_written_again() {
        let reader = StreamId { value: 1 };
        let recorded = ManagedMemoryId { value: 10 };
        let untouched = ManagedMemoryId { value: 11 };

        let mut errors = StreamErrors::default();
        errors.push_returned(error("capture"), [recorded, untouched]);

        errors.written([recorded]);
        assert!(
            errors.peek_unwritten(&[recorded], reader).is_empty(),
            "the relaunch wrote it, so a read of it is sound again"
        );
        assert_eq!(
            reasons(errors.peek_unwritten(&[untouched], reader)),
            ["capture"],
            "the buffers the relaunch did not write are still stale"
        );

        errors.written([untouched]);
        assert!(
            errors.peek_unwritten(&[untouched], reader).is_empty(),
            "with nothing left to answer for, the entry is gone"
        );
    }

    /// A flush is what ends an ordinary entry, so writing its buffers must not
    /// end it early.
    ///
    /// The error still has to reach the stream that caused it. Letting a
    /// relaunch strip its claim would leave a launch failure reported late and
    /// a read in between told nothing.
    #[test]
    fn writing_a_buffer_does_not_shorten_an_error_still_owed_to_a_stream() {
        let producer = StreamId { value: 1 };
        let reader = StreamId { value: 2 };
        let buffer = ManagedMemoryId { value: 10 };

        let mut errors = StreamErrors::default();
        errors.push_unwritten(producer, error("launch"), [buffer]);
        errors.push_returned(error("capture"), [ManagedMemoryId { value: 11 }]);

        errors.written([buffer]);

        assert_eq!(
            reasons(errors.peek_unwritten(&[buffer], reader)),
            ["launch"],
            "the producer's flush is this entry's bound, not somebody's relaunch"
        );
        assert_eq!(reasons(errors.take(producer)), ["launch"]);
    }

    /// Nothing drains a returned entry, so the cap is what keeps the queue from
    /// growing for the life of the process.
    ///
    /// Losing the oldest costs what a flush costs an ordinary entry — the
    /// buffers it named stop failing reads — so the newest are the ones kept.
    #[test]
    fn returned_entries_are_bounded_by_their_cap() {
        let reader = StreamId { value: 1 };
        let oldest = ManagedMemoryId { value: 0 };
        let newest = ManagedMemoryId {
            value: MAX_RETURNED,
        };

        let mut errors = StreamErrors::default();
        for value in 0..=MAX_RETURNED {
            errors.push_returned(error("capture"), [ManagedMemoryId { value }]);
        }

        assert_eq!(errors.returned(), MAX_RETURNED);
        assert!(
            errors.peek_unwritten(&[oldest], reader).is_empty(),
            "the oldest returned entry was dropped to stay under the cap"
        );
        assert_eq!(
            reasons(errors.peek_unwritten(&[newest], reader)),
            ["capture"],
            "the newest is the one worth keeping"
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
            errors.owned(),
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
            errors.entries[0].surfaced,
            Surfaced::AnyStream,
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
