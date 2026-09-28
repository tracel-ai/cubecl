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
pub struct DeviceCaptures(Arc<CapturePhases>);

/// Each capture's phase, and the counts read without the lock.
#[derive(Debug, Default)]
struct CapturePhases {
    /// How many logical streams are preparing or recording a capture: `owners.len()`.
    active: AtomicUsize,
    /// How many captures are recording: the [`CapturePhase::Recording`] entries of `owners`.
    recording: AtomicUsize,
    /// The logical streams preparing or recording a capture, with the phase each is in.
    owners: Mutex<Vec<(StreamId, CapturePhase)>>,
}

/// Where a capture is between `graph_prepare` and its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapturePhase {
    /// The warmup run, before the window opens.
    Preparing,
    /// The window is open: launches are recorded.
    Recording,
}

impl DeviceCaptures {
    /// Whether `stream` is preparing or recording a graph capture, from `graph_prepare` until
    /// the capture ends however it ends.
    pub fn is_capturing(&self, stream: StreamId) -> bool {
        self.0.active.load(Ordering::Acquire) > 0
            && self
                .0
                .owners
                .lock()
                .iter()
                .any(|(owner, _)| *owner == stream)
    }

    /// Whether any stream of the device records a graph.
    pub fn any_recording(&self) -> bool {
        self.0.recording.load(Ordering::Acquire) > 0
    }

    /// `owner` started the warmup run of a capture.
    pub fn prepare(&self, owner: StreamId) {
        self.update(|owners| {
            owners.retain(|(stream, _)| *stream != owner);
            owners.push((owner, CapturePhase::Preparing));
        });
    }

    /// The capture `owner` prepared opened its recording window.
    pub fn begin(&self, owner: StreamId) {
        self.update(|owners| {
            if let Some((_, phase)) = owners.iter_mut().find(|(stream, _)| *stream == owner) {
                *phase = CapturePhase::Recording;
            }
        });
    }

    /// The capture `owner` prepared is over, whichever phase it had reached.
    pub fn end(&self, owner: StreamId) {
        self.update(|owners| owners.retain(|(stream, _)| *stream != owner));
    }

    /// Apply `change` to the phases and republish the counts from them, so the counts can't
    /// drift from the phases they summarize.
    fn update(&self, change: impl FnOnce(&mut Vec<(StreamId, CapturePhase)>)) {
        let mut owners = self.0.owners.lock();
        change(&mut owners);
        let recording = owners
            .iter()
            .filter(|(_, phase)| *phase == CapturePhase::Recording)
            .count();
        self.0.recording.store(recording, Ordering::Release);
        self.0.active.store(owners.len(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: StreamId = StreamId { value: 7 };
    const OTHER: StreamId = StreamId { value: 8 };

    /// The counts follow the phases, whatever order the calls come in: a device that believed a
    /// capture still recorded would keep every stream from releasing a page.
    #[test]
    fn unmatched_calls_leave_the_counts_true() {
        let captures = DeviceCaptures::default();

        captures.end(OWNER);
        captures.begin(OWNER);
        assert!(
            !captures.any_recording(),
            "nothing prepared, nothing recording"
        );
        assert!(!captures.is_capturing(OWNER));

        captures.prepare(OWNER);
        captures.begin(OWNER);
        captures.end(OWNER);
        captures.end(OWNER);
        assert!(!captures.any_recording(), "a second end changes nothing");
        assert!(!captures.is_capturing(OWNER));

        captures.prepare(OWNER);
        captures.prepare(OTHER);
        captures.begin(OTHER);
        captures.end(OWNER);
        assert!(captures.any_recording(), "another capture still records");
        assert!(captures.is_capturing(OTHER));
        assert!(!captures.is_capturing(OWNER));
    }
}
