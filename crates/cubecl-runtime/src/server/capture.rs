use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{Arc, AtomicUsize, Mutex, Ordering};

/// The graph captures under way on one device: which logical streams are preparing or recording
/// one.
///
/// Every stream of the device updates it as its capture moves from `graph_prepare` through
/// `begin_capture` to `end_capture`, and every client of the device reads it, through the
/// [server utilities](super::ServerUtilities), without reaching the device thread. Code that
/// behaves differently while its stream captures a graph asks on every operation, eager ones
/// included, so the answer has to cost no round trip: with no capture under way it is one atomic
/// load.
#[derive(Debug, Clone, Default)]
pub struct DeviceCaptures(Arc<Captures>);

#[derive(Debug, Default)]
struct Captures {
    /// How many streams record a graph right now.
    recording: AtomicUsize,
    /// How many logical streams are preparing or recording a capture: `owners.len()`, readable
    /// without the lock.
    active: AtomicUsize,
    /// The logical streams preparing or recording a capture.
    owners: Mutex<Vec<StreamId>>,
}

impl DeviceCaptures {
    /// Whether `stream` is preparing or recording a graph capture, from `graph_prepare` until
    /// the capture ends or is abandoned.
    pub fn is_capturing(&self, stream: StreamId) -> bool {
        self.0.active.load(Ordering::Acquire) > 0 && self.0.owners.lock().contains(&stream)
    }

    /// Whether any stream of the device records a graph.
    pub fn any_recording(&self) -> bool {
        self.0.recording.load(Ordering::Acquire) > 0
    }

    /// `owner` started the warmup run of a capture.
    pub fn prepare(&self, owner: StreamId) {
        let mut owners = self.0.owners.lock();
        owners.push(owner);
        self.0.active.store(owners.len(), Ordering::Release);
    }

    /// A stream opened its recording window.
    pub fn begin(&self) {
        self.0.recording.fetch_add(1, Ordering::AcqRel);
    }

    /// The capture `owner` prepared is over, whether it `recorded` a window or never opened one.
    pub fn end(&self, owner: StreamId, recorded: Recorded) {
        if let Recorded::Yes = recorded {
            self.0.recording.fetch_sub(1, Ordering::AcqRel);
        }
        let mut owners = self.0.owners.lock();
        if let Some(pos) = owners.iter().position(|stream| *stream == owner) {
            owners.swap_remove(pos);
        }
        self.0.active.store(owners.len(), Ordering::Release);
    }
}

/// Whether a capture that ends had opened its recording window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recorded {
    /// The window opened: the stream counted as recording.
    Yes,
    /// The capture ended while still preparing.
    No,
}
