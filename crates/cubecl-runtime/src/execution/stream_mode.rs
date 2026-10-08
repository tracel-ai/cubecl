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
    /// [`CompileAndAutotune`](super::ProcessMode::CompileAndAutotune),
    /// queued for a batch otherwise — and the launches are discarded.
    Discard,
}

impl StreamMode {
    /// Its slot in a [`StreamModeRegistry`]'s live counts.
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

/// Sets the mode of one client's stream on its device while it lives,
/// restored on drop: what a measurement opens, so its launches run whatever
/// the process mode discards.
///
/// Keyed on the device and the stream the client's launches go out on — a
/// client bound to a stream of its own does not follow the thread's — so it
/// holds wherever those launches are issued from, a task resumed on another
/// thread included. Overrides of one stream nest, the newest deciding, and
/// may drop in any order.
///
/// Only the client's stream follows it: a launch another client issues on
/// another stream — a candidate holding a client bound elsewhere — keeps its
/// own stream's mode, and under a process mode that discards is discarded,
/// so a measurement launches through the client it switched.
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

/// The live [`StreamModeOverride`]s: what sets each stream's mode, and how
/// many set each mode, so a launch looks them up only when one sets the mode
/// its process mode does not — a measurement under no override, which sets
/// the mode every stream already has, costs other launches nothing.
pub(crate) struct StreamModeRegistry {
    /// The live overrides, oldest first.
    entries: Mutex<Vec<StreamModeEntry>>,
    /// How many live overrides set each mode, by [`StreamMode::index`].
    live: [AtomicUsize; 2],
    /// The id the next override takes.
    next_id: AtomicUsize,
}

/// The process's stream modes.
pub(crate) static STREAM_MODES: StreamModeRegistry = StreamModeRegistry {
    entries: Mutex::new(Vec::new()),
    live: [AtomicUsize::new(0), AtomicUsize::new(0)],
    next_id: AtomicUsize::new(0),
};

impl StreamModeRegistry {
    /// Put `stream` in `mode` until the guard it hands back drops.
    pub(crate) fn set(
        &'static self,
        mode: StreamMode,
        stream: ServiceStream,
    ) -> StreamModeOverride {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock();
        entries.push(StreamModeEntry { id, stream, mode });
        self.live[mode.index()].fetch_add(1, Ordering::Release);
        StreamModeOverride { id, mode }
    }

    /// `stream`'s mode now: what the newest live override sets for it, or
    /// `default` when none does.
    pub(crate) fn mode(&self, stream: ServiceStream, default: StreamMode) -> StreamMode {
        // An override setting the default changes nothing, unless it is
        // newer than one setting the other mode on the same stream, and then
        // there is one of those to look for.
        if self.live[default.opposite().index()].load(Ordering::Acquire) == 0 {
            return default;
        }
        self.entries
            .lock()
            .iter()
            .rev()
            .find(|entry| entry.stream == stream)
            .map_or(default, |entry| entry.mode)
    }

    fn unset(&self, id: usize, mode: StreamMode) {
        let mut entries = self.entries.lock();
        entries.retain(|entry| entry.id != id);
        self.live[mode.index()].fetch_sub(1, Ordering::Release);
    }
}

impl StreamModeOverride {
    /// Put `client`'s stream on its device in `mode` until the guard drops.
    pub fn new(mode: StreamMode, client: &Client) -> Self {
        STREAM_MODES.set(mode, client.service_stream())
    }
}

impl Drop for StreamModeOverride {
    fn drop(&mut self) {
        STREAM_MODES.unset(self.id, self.mode);
    }
}
