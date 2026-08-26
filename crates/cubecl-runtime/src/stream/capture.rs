//! The stream-side graph-capture lifecycle, shared by every backend with
//! graph support (see [`ComputeServer::graph_prepare`](crate::server::ComputeServer::graph_prepare)).

use crate::metadata_cache::CacheMode;
use crate::server::ServerError;
use cubecl_environment::stream::StreamId;

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
/// Both active states carry the logical stream that opened the capture. Several
/// logical streams share one backend stream, so "the capture owns this stream
/// for its window" only holds if the window remembers whose it is: an error
/// raised inside it belongs to the capture, not to whichever neighbour happens
/// to flush next (see [`StreamErrors`](super::StreamErrors)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCaptureState {
    /// No capture is prepared or recording.
    NoCapture,
    /// `graph_prepare` has armed the persistent pools for the warmup run;
    /// `begin_capture` may now open the window. Slices the warmup run reserves
    /// are retained by the memory manager's priming until `begin_capture` calls
    /// [`capture_priming_end`](crate::memory_management::MemoryManagement::capture_priming_end),
    /// so the pool ends up owning the capture run's full working set.
    Prepare {
        /// The logical stream that prepared the capture.
        owner: StreamId,
    },
    /// Launches are being recorded into a graph instead of executing. On a
    /// hardware-graph backend (CUDA, HIP) a host sync issued now aborts the
    /// driver capture, so the execution path defers fenced flushes until
    /// `end_capture`. A software-graph backend (wgpu) has no driver capture to
    /// abort and instead refuses the operations it cannot record: a read, sync
    /// or profile fails on the spot, while a write is rejected lazily — queued
    /// as an error that fails `end_capture`, since a graph missing an operation
    /// is worse than a late diagnostic.
    Capture {
        /// The logical stream recording the capture, which the errors raised
        /// inside the window belong to.
        owner: StreamId,
    },
}

/// What [`end`](StreamCaptureState::end) found when it closed the window: the
/// caller's own capture, or one belonging to a logical stream that never came
/// back to close it.
///
/// Both close the window. Only the owner gets a graph out of it — the errors
/// raised inside the window are queued for the owner, so sealing a recording
/// for a caller that cannot see them would hand back a graph silently missing
/// whatever they rejected.
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
        /// The logical stream that opened the window, whose queued errors the
        /// backend drains along with it — nothing is left waiting on a stream
        /// that may never flush again.
        owner: StreamId,
    },
}

impl CaptureEnd {
    /// The logical stream the window belonged to, and so the one whose errors
    /// the backend surfaces for it.
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
}

impl StreamCaptureState {
    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort (or is rejected
    /// by) the capture.
    pub fn is_recording(&self) -> bool {
        matches!(self, StreamCaptureState::Capture { .. })
    }

    /// Whether a capture is prepared or recording — the whole window during
    /// which the stream is not free to serve other work.
    pub fn is_active(&self) -> bool {
        !matches!(self, StreamCaptureState::NoCapture)
    }

    /// The logical stream this capture belongs to, `None` outside a window.
    pub fn owner(&self) -> Option<StreamId> {
        match self {
            StreamCaptureState::NoCapture => None,
            StreamCaptureState::Prepare { owner } | StreamCaptureState::Capture { owner } => {
                Some(*owner)
            }
        }
    }

    /// The [`CacheMode`] the metadata info cache should run in at this lifecycle
    /// position. Both while a graph is being *prepared* (warmup, which primes
    /// the cache) and while it is being *recorded* the cache runs in
    /// [`CacheMode::Capture`] — caching every buffer and invalidating none — so
    /// the capture window finds every info buffer warm and drops none out from
    /// under a recorded launch. Normal operation uses [`CacheMode::Normal`].
    pub fn cache_mode(&self) -> CacheMode {
        match self {
            StreamCaptureState::NoCapture => CacheMode::Normal,
            StreamCaptureState::Prepare { .. } | StreamCaptureState::Capture { .. } => {
                CacheMode::Capture
            }
        }
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
        match self {
            StreamCaptureState::NoCapture => {
                *self = StreamCaptureState::Prepare { owner };
                Ok(())
            }
            StreamCaptureState::Prepare { .. } => Err(ServerError::graph_state(
                "graph_prepare: a graph capture is already prepared on this stream",
            )),
            StreamCaptureState::Capture { .. } => Err(ServerError::graph_state(
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
    /// # Errors
    ///
    /// Fails when [`prepare`](Self::prepare) has not run — the persistent pools
    /// have to be primed by a warmup run first — or when a capture is already
    /// recording. The state is left untouched.
    pub fn begin(&mut self) -> Result<(), ServerError> {
        match self {
            StreamCaptureState::Prepare { owner } => {
                *self = StreamCaptureState::Capture { owner: *owner };
                Ok(())
            }
            StreamCaptureState::NoCapture => Err(ServerError::graph_state(
                "begin_capture: call graph_prepare before starting a capture",
            )),
            StreamCaptureState::Capture { .. } => Err(ServerError::graph_state(
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
        match self {
            StreamCaptureState::Capture { owner } => {
                let owner = *owner;
                *self = StreamCaptureState::NoCapture;
                Ok(match owner == caller {
                    true => CaptureEnd::Owned { owner },
                    false => CaptureEnd::Abandoned { owner },
                })
            }
            StreamCaptureState::NoCapture | StreamCaptureState::Prepare { .. } => {
                Err(ServerError::graph_state(
                    "end_capture: no graph capture is recording on this stream",
                ))
            }
        }
    }

    /// Return to `NoCapture` from anywhere, for the failure path of a
    /// transition's own work: the window never opened, so the stream must be
    /// left fully usable and re-capturable rather than stuck arming its
    /// persistent pools forever. Unlike [`end`](Self::end) this asserts
    /// nothing, because the state it is recovering from is precisely the one
    /// that could not be completed.
    pub fn abort(&mut self) {
        *self = StreamCaptureState::NoCapture;
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
        let mut state = StreamCaptureState::NoCapture;

        assert!(state.begin().is_err(), "a capture must be prepared first");
        assert!(state.end(OWNER).is_err(), "nothing is recording yet");
        assert_eq!(state, StreamCaptureState::NoCapture);

        state.prepare(OWNER).unwrap();
        assert_eq!(state, StreamCaptureState::Prepare { owner: OWNER });
        assert!(state.prepare(OWNER).is_err(), "one prepare per capture");
        assert!(state.end(OWNER).is_err(), "the window never opened");

        state.begin().unwrap();
        assert_eq!(state, StreamCaptureState::Capture { owner: OWNER });
        assert!(state.begin().is_err(), "captures may not overlap");
        assert!(state.prepare(OWNER).is_err(), "captures may not overlap");

        assert_eq!(
            state.end(OWNER).unwrap(),
            CaptureEnd::Owned { owner: OWNER }
        );
        assert_eq!(state, StreamCaptureState::NoCapture);
    }

    /// The window remembers whose it is from end to end, so the errors raised
    /// inside it can be queued for the stream that opened it rather than for
    /// whichever neighbour flushes the shared backend stream next.
    #[test]
    fn the_window_carries_its_owner() {
        let mut state = StreamCaptureState::NoCapture;
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
    /// A neighbour sealing it would hand back a recording while the errors
    /// raised inside stay queued for an owner the neighbour cannot name — the
    /// graph silently missing whatever those errors rejected.
    #[test]
    fn only_the_stream_that_opened_a_capture_may_seal_it() {
        let neighbour = StreamId { value: 8 };

        let mut state = StreamCaptureState::NoCapture;
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

        let mut state = StreamCaptureState::NoCapture;
        state.prepare(OWNER).unwrap();
        state.begin().unwrap();

        assert!(state.end(neighbour).unwrap().is_abandoned());
        assert_eq!(state, StreamCaptureState::NoCapture);
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

    /// A rejected transition leaves the stream exactly as it was, so a caller
    /// that miss orders a call can recover by issuing the right one — the
    /// property `wgpu_graph_lifecycle_state_errors` defends end to end.
    #[test]
    fn a_rejected_transition_changes_nothing() {
        let mut state = StreamCaptureState::Prepare { owner: OWNER };
        assert!(state.prepare(OWNER).is_err());
        assert_eq!(state, StreamCaptureState::Prepare { owner: OWNER });
        state.begin().unwrap();
    }

    /// `abort` recovers from a window that failed to open, from either of the
    /// states a backend can be holding when that happens.
    #[test]
    fn abort_recovers_a_window_that_never_opened() {
        for state in [
            StreamCaptureState::Prepare { owner: OWNER },
            StreamCaptureState::Capture { owner: OWNER },
        ] {
            let mut state = state;
            state.abort();
            assert_eq!(state, StreamCaptureState::NoCapture);
            state.prepare(OWNER).expect("the stream is re-capturable");
        }
    }

    /// The cache runs in capture mode for the *whole* prepare → record window,
    /// not just while recording: warmup is what makes the recorded launches hit
    /// warm info buffers, and an entry evicted between the two would be one a
    /// recorded launch dropped out from under itself.
    #[test]
    fn the_cache_captures_across_the_whole_window() {
        assert_eq!(
            StreamCaptureState::NoCapture.cache_mode(),
            CacheMode::Normal
        );
        assert_eq!(
            StreamCaptureState::Prepare { owner: OWNER }.cache_mode(),
            CacheMode::Capture
        );
        assert_eq!(
            StreamCaptureState::Capture { owner: OWNER }.cache_mode(),
            CacheMode::Capture
        );
    }
}
