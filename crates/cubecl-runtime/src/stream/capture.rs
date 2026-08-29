//! The stream-side graph-capture lifecycle, shared by every backend with
//! graph support (see [`ComputeServer::graph_prepare`](crate::server::ComputeServer::graph_prepare)).

use crate::metadata_cache::CacheMode;
use crate::server::{BufferBinding, ServerError};
use alloc::format;
use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum StreamCaptureState {
    /// No capture is prepared or recording.
    #[default]
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
    /// or profile fails on the spot, while a write is rejected lazily — the
    /// owner's own write dooms the recording so `end_capture` refuses to seal
    /// it, since a graph missing an operation is worse than a late diagnostic.
    Capture {
        /// The logical stream recording the capture, which the errors raised
        /// inside the window belong to.
        owner: StreamId,
    },
}

/// What [`StreamCapture::end`] found when it closed the window: the
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

/// The graph capture of one pooled backend stream: where it sits in the
/// lifecycle, and the memory its recorded launches were given.
///
/// The two travel together because neither is meaningful alone. A launch is
/// remembered only while the window is recording, and what it remembers is
/// only ever read once the window closes — so pairing them
/// makes "buffers accumulate inside a window and nowhere else" a property of
/// the type rather than a rule each backend has to keep.
///
/// Why the memory is worth keeping: a replay runs every recorded launch or
/// none, so a replay that fails to enqueue leaves all of them exactly as they
/// were, and so does a capture that never seals. Either way a later read of one
/// has to fail rather than copy out bytes nothing wrote — which needs the list
/// the launches themselves no longer hold, as bindings, so the failure can be
/// tainted onto the allocations they resolve to.
#[derive(Debug, Default)]
pub struct StreamCapture {
    state: StreamCaptureState,
    recorded: Vec<BufferBinding>,
    /// The host memory the recorded copies read from, held while the window
    /// is open and handed to the graph it seals into. A recorded memcpy node
    /// keeps the raw host pointer, so the bytes must live exactly as long as
    /// the graph — whatever kind of allocation they are, pinned-pool slice or
    /// plain heap. A window that never seals drops them here: its copies
    /// never ran and now never will.
    retained_host: Vec<Bytes>,
    /// The failure that doomed the window, if one did — work inside it failed
    /// or was skipped, so the recording is missing an operation. Sealing it
    /// would hand back a graph silently missing that work, and the replay
    /// contract has the caller write fresh inputs before each replay,
    /// clearing the very taint that would explain the hole — so the window is
    /// doomed instead, and `end_capture` refuses to seal it.
    failed: Option<ServerError>,
}

impl StreamCapture {
    /// Remember the memory a launch was given, when the stream is recording.
    ///
    /// A no-op outside a window, where a launch that fails taints its own
    /// buffers on the spot and there is no graph to answer for them later.
    pub fn record(&mut self, buffers: impl IntoIterator<Item = BufferBinding>) {
        if self.state.is_recording() {
            self.recorded.extend(buffers);
        }
    }

    /// The memory of the capture that just closed, each claim named once —
    /// the same buffer comes back once per recorded launch that was given it.
    ///
    /// Deduplicated by [`claim_key`](BufferBinding::claim_key), because the
    /// taint bookkeeping is range-exact and this list is what gets claimed
    /// and released: two tensors carved from one batched allocation are two
    /// claims, and collapsing them to their shared memory id would leave
    /// every sibling but one unclaimed on a refusal and unreleased on a
    /// replay.
    pub fn take_recorded(&mut self) -> Vec<BufferBinding> {
        let mut recorded = core::mem::take(&mut self.recorded);
        recorded.sort_unstable_by_key(|binding| binding.claim_key());
        recorded.dedup_by_key(|binding| binding.claim_key());
        recorded
    }

    /// Doom the recording: work inside the window failed or was skipped —
    /// see [`Self::take_failure`]. The first failure wins, and a stream that
    /// is not recording has no window to doom.
    pub fn fail(&mut self, error: ServerError) {
        if self.state.is_recording() && self.failed.is_none() {
            self.failed = Some(error);
        }
    }

    /// Keep `bytes` alive for the graph this recording seals into: a
    /// recorded copy holds their raw pointer and re-reads them on every
    /// replay, so they must not return to any pool or allocator while the
    /// graph lives. Handed to the graph by [`take_retained_host`](Self::take_retained_host).
    ///
    /// Only meaningful while recording — outside a window the bytes belong
    /// in the drop queue, whose fence knows when the device is done with
    /// them.
    pub fn retain_host(&mut self, bytes: Bytes) {
        debug_assert!(
            self.state.is_recording(),
            "host bytes are the window's to retain only while it records"
        );
        self.retained_host.push(bytes);
    }

    /// The host memory the window's recorded copies read from, taken as it
    /// closes — onto the graph when it seals, or to be dropped when it does
    /// not, since a recording that never becomes a graph never runs them.
    pub fn take_retained_host(&mut self) -> Vec<Bytes> {
        core::mem::take(&mut self.retained_host)
    }

    /// The failure that doomed this window, taken as it closes. `Some` means
    /// the recording is missing at least one operation and must not seal.
    pub fn take_failure(&mut self) -> Option<ServerError> {
        self.failed.take()
    }

    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort the capture, and
    /// during which a neighbour sharing the pooled stream is refused.
    pub fn is_recording(&self) -> bool {
        self.state.is_recording()
    }

    /// Whether a capture is prepared or recording — the span over which the
    /// pooled stream is committed to one logical stream's window.
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }

    /// The logical stream that opened the window, while one is open. `None`
    /// outside a capture, which is what distinguishes a neighbour's operation
    /// from the owner's.
    pub fn owner(&self) -> Option<StreamId> {
        self.state.owner()
    }

    /// How the metadata caches behave for this stream right now: a window
    /// pins what it builds, so a replay finds the same entries it recorded
    /// against.
    pub fn cache_mode(&self) -> CacheMode {
        self.state.cache_mode()
    }

    /// Arm the persistent pools for the warmup run; [`begin`](Self::begin)
    /// may open the window afterwards. A capture starts from an empty
    /// recording, so a window that was abandoned mid-flight cannot leak its
    /// buffers into the next one.
    ///
    /// # Errors
    ///
    /// Fails when a capture is already prepared or recording, leaving both the
    /// state and the recording untouched.
    pub fn prepare(&mut self, owner: StreamId) -> Result<(), ServerError> {
        self.state.prepare(owner)?;
        self.recorded.clear();
        self.retained_host.clear();
        self.failed = None;
        Ok(())
    }

    /// Open the window: launches from here until [`end`](Self::end) are
    /// recorded rather than executed.
    ///
    /// # Errors
    ///
    /// Fails when no capture is prepared, or one is already recording.
    pub fn begin(&mut self) -> Result<(), ServerError> {
        self.state.begin()
    }

    /// Close the window, saying whether `caller` owned it — see
    /// [`CaptureEnd`]. The recording survives the transition for
    /// [`take_recorded`](Self::take_recorded) to collect.
    ///
    /// # Errors
    ///
    /// Fails when no capture is recording, leaving the state untouched.
    pub fn end(&mut self, caller: StreamId) -> Result<CaptureEnd, ServerError> {
        self.state.end(caller)
    }

    /// Give up a prepared capture that never opened, restoring the stream to
    /// no-capture. Whatever it had recorded or retained goes with it.
    pub fn abort(&mut self) {
        self.state.abort();
        self.recorded.clear();
        self.retained_host.clear();
        self.failed = None;
    }
}

impl StreamCaptureState {
    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort (or is rejected
    /// by) the capture.
    pub(crate) fn is_recording(&self) -> bool {
        matches!(self, StreamCaptureState::Capture { .. })
    }

    /// Whether a capture is prepared or recording — the whole window during
    /// which the stream is not free to serve other work.
    pub(crate) fn is_active(&self) -> bool {
        !matches!(self, StreamCaptureState::NoCapture)
    }

    /// The logical stream this capture belongs to, `None` outside a window.
    pub(crate) fn owner(&self) -> Option<StreamId> {
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
    pub(crate) fn cache_mode(&self) -> CacheMode {
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
    pub(crate) fn prepare(&mut self, owner: StreamId) -> Result<(), ServerError> {
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
    pub(crate) fn begin(&mut self) -> Result<(), ServerError> {
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
    pub(crate) fn end(&mut self, caller: StreamId) -> Result<CaptureEnd, ServerError> {
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
    pub(crate) fn abort(&mut self) {
        *self = StreamCaptureState::NoCapture;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryId;
    use crate::server::Handle;

    const OWNER: StreamId = StreamId { value: 7 };

    /// A distinct buffer per call, on the owner's stream.
    fn buffer() -> BufferBinding {
        Handle::new(OWNER, 8).binding()
    }

    fn ids(bindings: &[BufferBinding]) -> Vec<ManagedMemoryId> {
        bindings.iter().map(|binding| binding.memory.id()).collect()
    }

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

    /// The window remembers whose it is from end to end, so a failure raised
    /// inside it dooms the capture that was recording rather than whichever
    /// neighbour happens to be sharing the backend stream.
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
    /// A neighbour sealing it would hand back a recording built from a window
    /// it never watched — the graph silently missing whatever the failures
    /// raised inside it rejected.
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

    /// A graph answers for every buffer its launches were given, and names each
    /// one once however many launches shared it.
    ///
    /// A failed replay runs none of the recorded launches, so the list is what
    /// a later read of any of those buffers fails on. Repeats would make that
    /// list grow with the length of the capture rather than with the working
    /// set — a chat step records hundreds of launches over the same handful of
    /// weights.
    #[test]
    fn a_capture_names_each_buffer_its_launches_were_given_once() {
        let (a, b, c) = (buffer(), buffer(), buffer());

        let mut capture = StreamCapture::default();
        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.record([a.clone(), b.clone()]);
        capture.record([b.clone(), c.clone()]);
        capture.record([a.clone()]);

        assert_eq!(
            capture.end(OWNER).unwrap(),
            CaptureEnd::Owned { owner: OWNER }
        );
        assert_eq!(ids(&capture.take_recorded()), ids(&[a, b, c]));
        assert!(
            capture.take_recorded().is_empty(),
            "the recording moves onto the graph, it is not left on the stream"
        );
    }

    /// Outside a window a launch answers for its own buffers as it fails, so
    /// nothing is remembered for a graph that will never exist.
    ///
    /// A stream that accumulated ids while not recording would grow one entry
    /// per launch for the life of the process, and hand the next capture a list
    /// of buffers it never touched.
    #[test]
    fn a_launch_outside_a_window_is_not_recorded() {
        let (before, warmup, recorded) = (buffer(), buffer(), buffer());
        let mut capture = StreamCapture::default();

        capture.record([before]);
        capture.prepare(OWNER).unwrap();
        // Prepared is the warmup run: it executes rather than records.
        capture.record([warmup]);

        capture.begin().unwrap();
        capture.record([recorded.clone()]);
        capture.end(OWNER).unwrap();

        assert_eq!(ids(&capture.take_recorded()), ids(&[recorded]));
    }

    /// A batched allocation is carved into sibling tensors sharing one memory
    /// id and nothing else, and the taint bookkeeping is range-exact — so the
    /// write set keeps every range. Collapsed to the id, a refusal would
    /// claim (and a replay release) only whichever sibling survived the
    /// dedup, leaving the others readable with stale bytes or unreadable with
    /// clean ones.
    #[test]
    fn a_capture_names_every_range_of_a_batched_allocation() {
        let handle = Handle::new(OWNER, 8);
        let mut front = handle.clone().binding();
        front.offset_end = Some(4);
        let mut back = handle.clone().binding();
        back.offset_start = Some(4);
        assert_eq!(
            front.memory.id(),
            back.memory.id(),
            "one allocation carved in two is the case under test"
        );

        let mut capture = StreamCapture::default();
        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.record([front.clone(), back.clone()]);
        // The same range again, from a second launch given the same tensor:
        // still one claim.
        capture.record([front.clone()]);
        capture.end(OWNER).unwrap();

        let recorded = capture.take_recorded();
        let keys: Vec<_> = recorded.iter().map(|binding| binding.claim_key()).collect();
        assert_eq!(
            keys,
            alloc::vec![front.claim_key(), back.claim_key()],
            "both siblings survive, each named once"
        );
    }

    /// The host bytes a recorded copy reads from live with the window: taken
    /// once as it closes — onto the graph, when one seals — and dropped with
    /// an aborted window, whose copies never ran and now never will.
    #[test]
    fn a_window_owns_the_host_bytes_its_copies_read() {
        let mut capture = StreamCapture::default();
        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.retain_host(Bytes::from_bytes_vec(alloc::vec![7u8; 4]));
        capture.end(OWNER).unwrap();

        assert_eq!(capture.take_retained_host().len(), 1);
        assert!(
            capture.take_retained_host().is_empty(),
            "taken means moved onto the graph, not copied"
        );

        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.retain_host(Bytes::from_bytes_vec(alloc::vec![7u8; 4]));
        capture.abort();
        capture.prepare(OWNER).unwrap();
        assert!(
            capture.take_retained_host().is_empty(),
            "an aborted window keeps nothing alive"
        );
    }

    /// A window that never sealed leaves nothing behind for the next one.
    ///
    /// The stream is re-capturable after an abort or an abandoned end, and a
    /// second capture that inherited the first one's buffers would pin memory
    /// its own launches never saw — failing reads that had nothing to do with
    /// it, on a graph that outlives the mistake.
    #[test]
    fn a_new_capture_starts_from_an_empty_recording() {
        let neighbour = StreamId { value: 8 };

        let (aborted, abandoned) = (buffer(), buffer());
        let mut capture = StreamCapture::default();
        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.record([aborted]);
        capture.abort();

        capture.prepare(OWNER).unwrap();
        capture.begin().unwrap();
        capture.record([abandoned.clone()]);
        assert!(capture.end(neighbour).unwrap().is_abandoned());
        assert_eq!(ids(&capture.take_recorded()), ids(&[abandoned]));

        capture.prepare(neighbour).unwrap();
        capture.begin().unwrap();
        assert!(
            capture.take_recorded().is_empty(),
            "the abandoned window's buffers are not this capture's to answer for"
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
