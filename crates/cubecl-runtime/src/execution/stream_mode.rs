use super::base::DeviceStream;
use crate::client::Client;
use alloc::vec::Vec;
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

/// Whether a stream's launches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// They run.
    Execute,
    /// Their kernels compile — now under
    /// [`CompileAndAutotune`](super::ExecutionPolicy::CompileAndAutotune), queued for
    /// a batch otherwise — and they are dropped.
    Compile,
}

/// Sets the mode of one client's stream on its device while it lives,
/// restored on drop: what a measurement opens, so its launches run whatever
/// the policy drops.
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
/// measurement runs executes too, rather than being dropped.
#[derive(Debug)]
pub struct StreamModeOverride {
    id: usize,
    mode: StreamMode,
}

/// One live [`StreamModeOverride`].
#[derive(Debug)]
struct StreamModeEntry {
    id: usize,
    on: DeviceStream,
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

impl StreamMode {
    /// Its slot in [`LIVE_STREAM_MODES`].
    fn index(self) -> usize {
        match self {
            Self::Execute => 0,
            Self::Compile => 1,
        }
    }
}

impl StreamModeOverride {
    /// Put `client`'s stream on its device in `mode` until the guard drops.
    pub fn new(mode: StreamMode, client: &Client) -> Self {
        Self::on(
            mode,
            DeviceStream {
                service: client.service_id(),
                stream: client.stream_id(),
            },
        )
    }

    pub(super) fn on(mode: StreamMode, on: DeviceStream) -> Self {
        let id = NEXT_STREAM_MODE.fetch_add(1, Ordering::Relaxed);
        let mut modes = STREAM_MODES.lock();
        modes.push(StreamModeEntry { id, on, mode });
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

/// The mode the newest live override sets for `on`, or `default` when none
/// does.
pub(super) fn stream_mode(on: DeviceStream, default: StreamMode) -> StreamMode {
    let other = match default {
        StreamMode::Execute => StreamMode::Compile,
        StreamMode::Compile => StreamMode::Execute,
    };
    // An override setting the default changes nothing, unless it is newer
    // than one setting the other mode on the same stream, and then there is
    // one of those to look for.
    if LIVE_STREAM_MODES[other.index()].load(Ordering::Acquire) == 0 {
        return default;
    }
    STREAM_MODES
        .lock()
        .iter()
        .rev()
        .find(|entry| entry.on == on)
        .map_or(default, |entry| entry.mode)
}
