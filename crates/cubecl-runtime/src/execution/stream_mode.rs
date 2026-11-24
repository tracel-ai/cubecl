use super::STREAM_MODES;
use crate::client::Client;
use cubecl_common::device::ServiceId;
use cubecl_environment::stream::StreamId;

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

/// One stream of one device's service: what a [`StreamModeOverride`] sets
/// the mode of. Stream ids are the process's, not a device's, so a
/// measurement on one device leaves the same stream of every other device
/// alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceStream {
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

impl StreamModeOverride {
    /// Put `client`'s stream on its device in `mode` until the guard drops.
    pub fn new(mode: StreamMode, client: &Client) -> Self {
        Self::on_stream(mode, client.service_stream())
    }
}

impl StreamModeOverride {
    /// Put `stream` in `mode` until the guard drops.
    pub(crate) fn on_stream(mode: StreamMode, stream: ServiceStream) -> Self {
        Self {
            id: STREAM_MODES.set(mode, stream),
            mode,
        }
    }
}

impl Drop for StreamModeOverride {
    fn drop(&mut self) {
        STREAM_MODES.unset(self.id, self.mode);
    }
}
