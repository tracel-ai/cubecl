use crate::client::Client;
use alloc::vec::Vec;
use cubecl_common::device::ServiceId;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

/// Whether a stream's launches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// They run.
    Execute,
    /// Their kernels compile — now under
    /// [`CompileAndAutotune`](super::ExecutionPolicy::CompileAndAutotune),
    /// queued for a batch otherwise — and the launches are discarded.
    Discard,
}

impl StreamMode {
    /// Its slot in [`LIVE_STREAM_MODES`].
    fn index(self) -> usize {
        match self {
            Self::Execute => 0,
            Self::Discard => 1,
        }
    }

    /// The other mode.
    fn opposite(self) -> Self {
        match self {
            Self::Execute => Self::Discard,
            Self::Discard => Self::Execute,
        }
    }
}

/// One stream of one device's service: what a [`StreamModeOverride`] sets
/// the mode of. Stream ids are the process's, not a device's, so a
/// measurement on one device leaves the same stream of every other device
/// alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServiceStream {
    /// The device's service, as its clients reach it.
    pub service: ServiceId,
    /// The stream on it.
    pub stream: StreamId,
}

impl ServiceStream {
    /// Its mode now: what the newest live override sets for it, or
    /// `default` when none does.
    pub(crate) fn mode(self, default: StreamMode) -> StreamMode {
        // An override setting the default changes nothing, unless it is
        // newer than one setting the other mode on the same stream, and then
        // there is one of those to look for.
        if LIVE_STREAM_MODES[default.opposite().index()].load(Ordering::Acquire) == 0 {
            return default;
        }
        STREAM_MODES
            .lock()
            .iter()
            .rev()
            .find(|entry| entry.stream == self)
            .map_or(default, |entry| entry.mode)
    }
}

/// Sets the mode of one client's stream on its device while it lives,
/// restored on drop: what a measurement opens, so its launches run whatever
/// the policy discards.
///
/// Keyed on the device and the stream the client's launches go out on — a
/// client bound to a stream of its own does not follow the thread's — so it
/// holds wherever those launches are issued from, a task resumed on another
/// thread included. Overrides of one stream nest, the newest deciding, and
/// may drop in any order.
///
/// Every launch on that stream of that device follows it, whichever thread
/// issues it. Where threads share a stream — `StreamPolicy::Single`, or a
/// build without per-thread streams — a launch another thread issues while a
/// measurement runs executes too, rather than being discarded.
#[derive(Debug)]
pub struct StreamModeOverride {
    id: usize,
    mode: StreamMode,
}

/// One live [`StreamModeOverride`].
#[derive(Debug)]
struct StreamModeEntry {
    id: usize,
    stream: ServiceStream,
    mode: StreamMode,
}

/// How many live [`StreamModeOverride`]s set each mode, by
/// [`StreamMode::index`]: a launch looks the overrides up only when one sets
/// the mode its policy does not, so a measurement under no override, which
/// sets the mode every stream already has, costs other launches nothing.
static LIVE_STREAM_MODES: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
/// The live overrides, oldest first.
static STREAM_MODES: Mutex<Vec<StreamModeEntry>> = Mutex::new(Vec::new());
/// The id the next override takes.
static NEXT_STREAM_MODE: AtomicUsize = AtomicUsize::new(0);

impl StreamModeOverride {
    /// Put `client`'s stream on its device in `mode` until the guard drops.
    pub fn new(mode: StreamMode, client: &Client) -> Self {
        Self::of_stream(mode, client.service_stream())
    }

    /// Put `stream` in `mode` until the guard drops.
    pub(crate) fn of_stream(mode: StreamMode, stream: ServiceStream) -> Self {
        let id = NEXT_STREAM_MODE.fetch_add(1, Ordering::Relaxed);
        let mut modes = STREAM_MODES.lock();
        modes.push(StreamModeEntry { id, stream, mode });
        LIVE_STREAM_MODES[mode.index()].fetch_add(1, Ordering::Release);
        Self { id, mode }
    }
}

impl Drop for StreamModeOverride {
    fn drop(&mut self) {
        let mut modes = STREAM_MODES.lock();
        modes.retain(|entry| entry.id != self.id);
        LIVE_STREAM_MODES[self.mode.index()].fetch_sub(1, Ordering::Release);
    }
}
