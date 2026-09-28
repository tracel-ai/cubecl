//! The graph-capture lifecycle of a device's streams, and the record of it every client reads.

use alloc::format;
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{Arc, AtomicUsize, Mutex, Ordering};

use super::ServerError;

/// The graph captures under way on one device.
///
/// [`ServerUtilities::init`](super::ServerUtilities::init) creates it with the read-only
/// [status](Self::status) the utilities hand to every client. The server creates one
/// [capture state](Self::stream) per backend stream from it, and only those states change what
/// it records: each publishes its own transitions, so the record can't drift from the streams.
#[derive(Debug, Clone, Default)]
pub struct DeviceCaptures(Arc<CaptureRecord>);

/// A read-only view of a device's [captures](DeviceCaptures).
///
/// Every client of the device reads it, through the [server utilities](super::ServerUtilities),
/// without reaching the device thread. It is asked before every launch that has to decide the
/// same way in a capture's warmup and its recording, so it can't cost a round trip: with no
/// capture under way on the device, it is one atomic load.
#[derive(Debug, Clone)]
pub struct CaptureStatus(Arc<CaptureRecord>);

/// Where a stream sits in the graph-capture lifecycle, and the only thing
/// allowed to move it. Capture is a strict `NoCapture → Prepare → Capture →
/// NoCapture` progression, driven by [`prepare`](Self::prepare),
/// [`begin`](Self::begin) and [`end`](Self::end); each rejects an out-of-order
/// call, so a capture can never start unprepared and two captures can never
/// overlap on one stream.
///
/// # One capture, one logical stream
///
/// The three calls have to come from the same logical stream. The window is
/// opened on the pooled stream that logical stream folds onto, and the launches
/// in between are recorded there — so a caller whose [`StreamId`] changes
/// half-way (an `.await` resuming on another thread under the default
/// `PerThread` policy, without `set_stream` pinning) has already split its
/// recording across two backend streams before it ever reaches `end`. Pin the
/// stream around a capture; [`end`](Self::end) treats a caller that is not the
/// owner as a window nobody is coming back for, and abandons it.
///
/// The transitions live here rather than in each backend server because the
/// rule is the same on every one of them — a backend supplies only the work a
/// transition brackets (arming its pools, opening the driver's capture), never
/// the ordering rule itself.
///
/// # What the neighbours pay
///
/// The window is held on a pooled stream, and logical streams fold onto those
/// with `id % max_streams` — so a capture costs every logical stream sharing
/// that slot, not just the one recording. On a software-graph backend a
/// neighbour's read, sync or profile is refused outright for the duration, and
/// its write is refused with the refusal landing on its own destinations; on a
/// hardware-graph backend a
/// neighbour's fenced flush is deferred until the window closes. None of that
/// is attributed to the capture, because a refusal is not a failure of the
/// capture: the neighbour asked for something this slot cannot do right now.
///
/// It is a real cost of folding, and the reason a capture is worth pinning to a
/// stream nothing else is scheduled on.
///
/// Both active states carry the logical stream that opened the capture. Several
/// logical streams share one backend stream, so "the capture owns this stream
/// for its window" only holds if the window remembers whose it is: an error
/// raised inside it dooms the capture, not whichever neighbour happens to be
/// using the slot.
#[derive(Debug)]
pub struct StreamCaptureState {
    phase: CapturePhase,
    /// The device's record, which every transition publishes the new phase to.
    device: Arc<CaptureRecord>,
    /// This state's entry in the device's record.
    entry: usize,
}

/// What [`StreamCaptureState::end`] found when it closed the window: the
/// caller's own capture, or one belonging to a logical stream that never came
/// back to close it.
///
/// Both close the window. Only the owner gets a graph out of it: the failures
/// raised inside the window doom the recording, and sealing it for a caller
/// that never saw them would hand back a graph silently missing whatever they
/// rejected.
///
/// Refusing a foreign caller outright is the worse trade. Several logical
/// streams share one pooled stream, and a window nobody closes rejects every
/// read, write and sync that lands on the slot while recording launches into a
/// graph no one can seal: the slot is lost for the life of the process. A
/// foreign `end_capture` is a caller whose [`StreamId`] moved out from under it
/// (see the type docs), which is exactly the case where the owner is gone — so
/// the window is torn down and the failure reported, rather than kept for an
/// owner that will never ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureEnd {
    /// The caller opened this window: its recording may be sealed into a graph.
    Owned {
        /// The logical stream that opened the window, which is the caller.
        owner: StreamId,
    },
    /// The window belonged to `owner`, not to the caller. It is closed, but
    /// there is no graph to hand back: the backend tears the recording down and
    /// reports, and a later `end_capture` from `owner` finds nothing recording.
    Abandoned {
        /// The logical stream that opened the window, which the report names
        /// so the caller can see whose recording was discarded.
        owner: StreamId,
    },
}

impl CaptureEnd {
    /// The logical stream the window belonged to.
    pub fn owner(&self) -> StreamId {
        match self {
            CaptureEnd::Owned { owner } | CaptureEnd::Abandoned { owner } => *owner,
        }
    }

    /// Whether the window was closed for a caller that did not own it, so its
    /// recording is torn down instead of sealed.
    pub fn is_abandoned(&self) -> bool {
        matches!(self, CaptureEnd::Abandoned { .. })
    }

    /// The report a caller gets for closing a window it did not open: why the
    /// recording was discarded, and then `doomed` — the failure that had
    /// already sunk the recording, if one had, so the caller learns both
    /// reasons rather than only the one that happened to be checked last.
    ///
    /// Only meaningful once [`is_abandoned`](Self::is_abandoned) says so; an
    /// owned window is the caller's to seal and has nothing to report.
    pub fn abandoned_error(&self, caller: StreamId, doomed: Option<ServerError>) -> ServerError {
        let mut errors = alloc::vec![ServerError::graph_state(format!(
            "end_capture: the capture belongs to logical stream {:?}, not to {caller:?}; it is \
             discarded rather than left recording on a stream both share",
            self.owner(),
        ))];
        errors.extend(doomed);

        ServerError::Several {
            errors,
            backtrace: BackTrace::capture(),
        }
    }
}

/// Where one backend stream is in the capture lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapturePhase {
    /// No capture is prepared or recording.
    NoCapture,
    /// `graph_prepare` has started the warmup run; `begin_capture` may now
    /// open the window. The pinned staging the warmup run reserves is
    /// primed (see `StreamCapture::prime` in cubecl-server) until the window opens.
    Prepare {
        /// The logical stream that prepared the capture.
        owner: StreamId,
    },
    /// Launches are being recorded into a graph instead of executing. On a
    /// hardware-graph backend (CUDA, HIP) a host sync issued now aborts the
    /// driver capture, so the execution path defers fenced flushes until
    /// `end_capture`. A software-graph backend (wgpu) has no driver capture to
    /// abort and instead refuses the operations it cannot record: a read, sync
    /// or profile fails on the spot, while a write is rejected lazily — the
    /// owner's own write dooms the recording so `end_capture` refuses to seal
    /// it, since a graph missing an operation is worse than a late diagnostic.
    Capture {
        /// The logical stream recording the capture, which the errors raised
        /// inside the window belong to.
        owner: StreamId,
    },
}

/// The phase of every backend stream of a device with a capture under way, and the counts read
/// without the lock.
#[derive(Debug, Default)]
struct CaptureRecord {
    /// How many streams are preparing or recording a capture: `streams.len()`.
    active: AtomicUsize,
    /// How many streams are recording: the recording entries of `streams`.
    recording: AtomicUsize,
    /// One entry per stream with a capture under way: its [entry](StreamCaptureState::entry)
    /// id, the logical stream the capture belongs to, and whether it records.
    streams: Mutex<Vec<(usize, StreamId, bool)>>,
    /// The next [entry](StreamCaptureState::entry) id.
    next_entry: AtomicUsize,
}

impl DeviceCaptures {
    /// The capture state of one more backend stream of the device.
    pub fn stream(&self) -> StreamCaptureState {
        StreamCaptureState {
            phase: CapturePhase::NoCapture,
            device: self.0.clone(),
            entry: self.0.next_entry.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// The read-only view of these captures, for everyone but the streams.
    pub fn status(&self) -> CaptureStatus {
        CaptureStatus(self.0.clone())
    }
}

impl CaptureStatus {
    /// Whether any stream of the device is preparing or recording a capture: when not, no
    /// logical stream is [capturing](Self::is_capturing), whichever it is.
    #[inline]
    pub fn any_active(&self) -> bool {
        self.0.active.load(Ordering::Acquire) > 0
    }

    /// Whether the logical `stream` is preparing or recording a graph capture, from
    /// `graph_prepare` until the capture ends however it ends.
    pub fn is_capturing(&self, stream: StreamId) -> bool {
        self.any_active()
            && self
                .0
                .streams
                .lock()
                .iter()
                .any(|(_, owner, _)| *owner == stream)
    }
}

impl StreamCaptureState {
    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort (or is rejected
    /// by) the capture.
    pub fn is_recording(&self) -> bool {
        matches!(self.phase, CapturePhase::Capture { .. })
    }

    /// Whether the warmup run is under way: prepared, and the window not open yet.
    pub fn is_preparing(&self) -> bool {
        matches!(self.phase, CapturePhase::Prepare { .. })
    }

    /// Whether a capture is prepared or recording — the whole window during
    /// which the stream is not free to serve other work.
    pub fn is_active(&self) -> bool {
        !matches!(self.phase, CapturePhase::NoCapture)
    }

    /// The logical stream this capture belongs to, `None` outside a window.
    pub fn owner(&self) -> Option<StreamId> {
        match self.phase {
            CapturePhase::NoCapture => None,
            CapturePhase::Prepare { owner } | CapturePhase::Capture { owner } => Some(owner),
        }
    }

    /// Whether any stream of the device records a graph, this one included.
    #[inline]
    pub fn any_recording(&self) -> bool {
        self.device.recording.load(Ordering::Acquire) > 0
    }

    /// `NoCapture → Prepare`, for `graph_prepare`. Call before arming the
    /// pools; the caller owns the arming, this owns the rule that it happens
    /// exactly once per capture.
    ///
    /// # Errors
    ///
    /// Fails when a capture is already prepared or already recording on this
    /// stream, leaving the state untouched — two captures may never overlap on
    /// one stream. The caller can retry after `end_capture`.
    pub fn prepare(&mut self, owner: StreamId) -> Result<(), ServerError> {
        match self.phase {
            CapturePhase::NoCapture => {
                self.move_to(CapturePhase::Prepare { owner });
                Ok(())
            }
            CapturePhase::Prepare { .. } => Err(ServerError::graph_state(
                "graph_prepare: a graph capture is already prepared on this stream",
            )),
            CapturePhase::Capture { .. } => Err(ServerError::graph_state(
                "graph_prepare: a graph capture is already recording on this stream",
            )),
        }
    }

    /// `Prepare → Capture`, for `begin_capture`. Call *before* the work that
    /// opens the window (ending the priming phase, starting the driver's
    /// capture) so a rejected call cannot run any of it: on a stream that is
    /// already recording, a drop-queue flush issued on the way to the rejection
    /// would abort the live capture.
    ///
    /// Since the state moves before that work, a backend whose window fails to
    /// open must undo it with [`abort`](Self::abort).
    ///
    /// Returns the logical stream the capture belongs to.
    ///
    /// # Errors
    ///
    /// Fails when [`prepare`](Self::prepare) has not run — the pools have to
    /// be warmed by a warmup run first — or when a capture is already
    /// recording. The state is left untouched.
    pub fn begin(&mut self) -> Result<StreamId, ServerError> {
        match self.phase {
            CapturePhase::Prepare { owner } => {
                self.move_to(CapturePhase::Capture { owner });
                Ok(owner)
            }
            CapturePhase::NoCapture => Err(ServerError::graph_state(
                "begin_capture: call graph_prepare before starting a capture",
            )),
            CapturePhase::Capture { .. } => Err(ServerError::graph_state(
                "begin_capture: a graph capture is already recording on this stream",
            )),
        }
    }

    /// `Capture → NoCapture`, for `end_capture`. Call before closing the
    /// window, so the stream leaves capture state even if sealing the graph
    /// then fails — a backend that returned an error with the state still set
    /// would wedge the stream in capture mode forever.
    ///
    /// A caller that does not own the window closes it all the same, as
    /// [`CaptureEnd::Abandoned`]: only the owner may *seal* a capture, but
    /// leaving the window open until an owner that may never come back closes
    /// it would wedge the pooled stream for every logical stream sharing it —
    /// see [`CaptureEnd`] for why that is the lesser of the two.
    ///
    /// # Errors
    ///
    /// Fails when no capture is recording (nothing prepared or started, or the
    /// capture already ended), leaving the state untouched — a stray
    /// `end_capture` must not close a window that was never opened.
    pub fn end(&mut self, caller: StreamId) -> Result<CaptureEnd, ServerError> {
        match self.phase {
            CapturePhase::Capture { owner } => {
                self.move_to(CapturePhase::NoCapture);
                Ok(match owner == caller {
                    true => CaptureEnd::Owned { owner },
                    false => CaptureEnd::Abandoned { owner },
                })
            }
            CapturePhase::NoCapture | CapturePhase::Prepare { .. } => {
                Err(ServerError::graph_state(
                    "end_capture: no graph capture is recording on this stream",
                ))
            }
        }
    }

    /// Return to `NoCapture` from anywhere, for the failure path of a
    /// transition's own work: the window never opened, so the stream must be
    /// left fully usable and re-capturable rather than stuck preparing
    /// forever. Unlike [`end`](Self::end) this asserts
    /// nothing, because the state it is recovering from is precisely the one
    /// that could not be completed.
    ///
    /// Returns the logical stream the abandoned capture belonged to, if one was under way.
    pub fn abort(&mut self) -> Option<StreamId> {
        let owner = self.owner();
        self.move_to(CapturePhase::NoCapture);
        owner
    }

    /// Enter `phase` and publish it to the device's record, in one step so the record can't
    /// miss a transition.
    fn move_to(&mut self, phase: CapturePhase) {
        self.phase = phase;
        self.device.publish(self.entry, phase);
    }
}

impl Drop for StreamCaptureState {
    fn drop(&mut self) {
        self.device.publish(self.entry, CapturePhase::NoCapture);
    }
}

impl CaptureRecord {
    /// Record that the stream with `entry` is now in `phase`, and recompute the counts from the
    /// entries under the lock, so they can't drift from the phases they summarize.
    fn publish(&self, entry: usize, phase: CapturePhase) {
        let mut streams = self.streams.lock();
        streams.retain(|(id, _, _)| *id != entry);
        match phase {
            CapturePhase::NoCapture => {}
            CapturePhase::Prepare { owner } => streams.push((entry, owner, false)),
            CapturePhase::Capture { owner } => streams.push((entry, owner, true)),
        }
        let recording = streams
            .iter()
            .filter(|(_, _, recording)| *recording)
            .count();
        self.recording.store(recording, Ordering::Release);
        self.active.store(streams.len(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: StreamId = StreamId { value: 7 };

    /// The ordering rule the three backends rely on: a capture cannot start
    /// unprepared, and two cannot overlap on one stream. A backend that could
    /// reach `Capture` without `Prepare` would record against pools no warmup
    /// primed, and every allocation the window then makes is one the graph
    /// replays against but nothing pins.
    #[test]
    fn transitions_follow_the_capture_order() {
        let mut state = DeviceCaptures::default().stream();

        assert!(state.begin().is_err(), "a capture must be prepared first");
        assert!(state.end(OWNER).is_err(), "nothing is recording yet");
        assert!(!state.is_active());

        state.prepare(OWNER).unwrap();
        assert!(state.is_preparing() && state.owner() == Some(OWNER));
        assert!(state.prepare(OWNER).is_err(), "one prepare per capture");
        assert!(state.end(OWNER).is_err(), "the window never opened");

        assert_eq!(state.begin().unwrap(), OWNER);
        assert!(state.is_recording() && state.owner() == Some(OWNER));
        assert!(state.begin().is_err(), "captures may not overlap");
        assert!(state.prepare(OWNER).is_err(), "captures may not overlap");

        assert_eq!(
            state.end(OWNER).unwrap(),
            CaptureEnd::Owned { owner: OWNER }
        );
        assert!(!state.is_active());
    }

    /// The window remembers whose it is from end to end, so a failure raised
    /// inside it dooms the capture that was recording rather than whichever
    /// neighbour happens to be sharing the backend stream.
    #[test]
    fn the_window_carries_its_owner() {
        let mut state = DeviceCaptures::default().stream();
        assert_eq!(state.owner(), None);

        state.prepare(OWNER).unwrap();
        assert_eq!(state.owner(), Some(OWNER));
        assert!(state.is_active(), "the window is open from prepare on");

        state.begin().unwrap();
        assert_eq!(state.owner(), Some(OWNER));

        assert_eq!(state.end(OWNER).unwrap().owner(), OWNER);
        assert_eq!(state.owner(), None);
        assert!(!state.is_active());
    }

    /// Only the stream that opened the window may seal it into a graph.
    ///
    /// A neighbour sealing it would hand back a recording built from a window
    /// it never watched — the graph silently missing whatever the failures
    /// raised inside it rejected.
    #[test]
    fn only_the_stream_that_opened_a_capture_may_seal_it() {
        let neighbour = StreamId { value: 8 };

        let mut state = DeviceCaptures::default().stream();
        state.prepare(OWNER).unwrap();
        state.begin().unwrap();

        assert_eq!(
            state.end(neighbour).unwrap(),
            CaptureEnd::Abandoned { owner: OWNER },
            "the window is not theirs to seal"
        );
    }

    /// A window its owner never closes must not hold the pooled stream, which
    /// every logical stream folded onto the slot shares.
    ///
    /// The owner's id can stop coming back — the thread that started the
    /// capture exits, or an `.await` resumes it elsewhere under `PerThread`. A
    /// window kept until that id returns rejects every read, write and sync on
    /// the slot forever, so a foreign `end` closes it and leaves the stream
    /// usable, reporting rather than sealing.
    #[test]
    fn a_capture_no_one_can_close_does_not_wedge_the_stream() {
        let neighbour = StreamId { value: 8 };

        let mut state = DeviceCaptures::default().stream();
        state.prepare(OWNER).unwrap();
        state.begin().unwrap();

        assert!(state.end(neighbour).unwrap().is_abandoned());
        assert!(!state.is_active());
        assert!(!state.is_active(), "the slot serves other work again");
        state
            .prepare(neighbour)
            .expect("the stream is re-capturable");

        // The owner coming back late finds nothing recording, rather than a
        // window it can still seal a graph out of.
        state.begin().unwrap();
        assert!(
            state.end(OWNER).unwrap().is_abandoned(),
            "the window it opened is long gone"
        );
    }

    /// What a caller learns from closing a window that was not theirs: whose
    /// it was, and whatever had already doomed the recording.
    ///
    /// The owner is the one piece of evidence the caller can act on — it names
    /// the stream whose recording was thrown away. The doomed reason travels
    /// with it because both are true at once, and reporting only the
    /// abandonment would hide a failure that had already made the recording
    /// unsealable.
    #[test]
    fn an_abandoned_window_reports_whose_it_was_and_what_doomed_it() {
        let caller = StreamId { value: 8 };
        let outcome = CaptureEnd::Abandoned { owner: OWNER };

        let error = outcome.abandoned_error(caller, Some(ServerError::graph_state("doomed")));

        let ServerError::Several { errors, .. } = &error else {
            panic!("an abandoned window reports several failures at once, got: {error:?}");
        };
        let reported = alloc::format!("{error}");
        assert!(
            reported.contains(&alloc::format!("{OWNER:?}"))
                && reported.contains(&alloc::format!("{caller:?}")),
            "the report has to name the window's owner and the caller refused it, got: {reported}"
        );
        assert_eq!(errors.len(), 2, "the doomed reason travels with it");
        assert!(
            alloc::format!("{}", errors[1]).contains("doomed"),
            "the explanation comes first, then what had already sunk it"
        );
    }

    /// A rejected transition leaves the stream exactly as it was, so a caller
    /// that miss orders a call can recover by issuing the right one — the
    /// property `wgpu_graph_lifecycle_state_errors` defends end to end.
    #[test]
    fn a_rejected_transition_changes_nothing() {
        let mut state = DeviceCaptures::default().stream();
        state.prepare(OWNER).unwrap();
        assert!(state.prepare(OWNER).is_err());
        assert!(state.is_preparing() && state.owner() == Some(OWNER));
        state.begin().unwrap();
    }

    /// `abort` recovers from a window that failed to open, from either of the
    /// states a backend can be holding when that happens.
    #[test]
    fn abort_recovers_a_window_that_never_opened() {
        for recorded in [false, true] {
            let mut state = DeviceCaptures::default().stream();
            state.prepare(OWNER).unwrap();
            if recorded {
                state.begin().unwrap();
            }
            assert_eq!(state.abort(), Some(OWNER));
            assert!(!state.is_active());
            state.prepare(OWNER).expect("the stream is re-capturable");
        }
    }

    /// The device's record follows each stream's own state, whatever the streams do: a record
    /// that believed no capture recorded while one did would let the neighbours release pages
    /// the recording uses.
    #[test]
    fn the_record_counts_every_recording_stream() {
        let device = DeviceCaptures::default();
        let status = device.status();
        let mut first = device.stream();
        let mut second = device.stream();

        first.prepare(OWNER).unwrap();
        first.begin().unwrap();
        second.prepare(OWNER).unwrap();
        assert!(
            first.any_recording(),
            "a second stream of the owner hides nothing"
        );
        second.begin().unwrap();
        first.end(OWNER).unwrap();
        assert!(
            second.any_recording(),
            "one stream ending leaves the other recording"
        );
        assert!(status.is_capturing(OWNER));

        drop(second);
        assert!(!first.any_recording(), "a dropped stream leaves the record");
        assert!(!status.is_capturing(OWNER));
        assert!(!status.any_active());
    }
}
