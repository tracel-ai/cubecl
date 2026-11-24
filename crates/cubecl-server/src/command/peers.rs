//! The memory of other devices one launch binds, and what the launch's outcome
//! means for it.
//!
//! A launch's own write scope settles the buffers of its device. Memory of
//! another device is settled here instead: its owner keeps the failures, so a
//! failure reaches it as a task on its own thread, and the [`PeerMemory`] both
//! devices hold is what a peer kernel checks before reading.

use crate::kernel::BufferIOAttr;
use crate::server::{BufferBinding, KernelArguments, PeerMemory, ServerError};
use alloc::vec;
use alloc::vec::Vec;

/// The memory of other devices one launch reads and writes.
///
/// Empty, and allocation-free, for a launch that binds none.
#[derive(Debug, Default)]
pub struct PeerLaunch {
    read: Vec<PeerMemory>,
    written: Vec<BufferBinding>,
}

impl PeerLaunch {
    /// The peer memory `bindings` reads and writes, by the compiled kernel's
    /// answer `io` when it kept one and the caller's declaration otherwise.
    pub fn new(bindings: &KernelArguments, io: Option<&[BufferIOAttr]>) -> Self {
        Self {
            read: bindings
                .peers_read(io)
                .map(|peer| peer.memory().clone())
                .collect(),
            written: bindings
                .peers_written(io)
                .map(|peer| peer.binding().clone())
                .collect(),
        }
    }

    /// The failure a peer input carries: work on its owner that should have
    /// written it failed. The launch must not run, as it would not with a
    /// failed input of its own device.
    pub fn input_failure(&self) -> Option<ServerError> {
        self.read.iter().find_map(PeerMemory::failure)
    }

    /// The launch did not run, or failed, with `error`: the peer memory it was
    /// going to write carries the failure, here for other peers and on its
    /// owner for a read there.
    pub fn failed(&self, error: &ServerError) {
        for binding in &self.written {
            let peer = shared(binding);
            peer.fail(error.clone());
            let written = vec![binding.clone()];
            let error = error.clone();
            // Not joined: the owner settles it in its own order, ahead of any
            // read submitted after the caller synced this device.
            drop(
                peer.owner()
                    .spawn(move |server| server.peer_wrote(written, Some(error))),
            );
        }
    }

    /// The launch was enqueued: the peer memory it writes no longer carries a
    /// failure an earlier writer left, here or on its owner.
    pub fn enqueued(&self) {
        for binding in &self.written {
            let peer = shared(binding);
            if peer.repair() {
                let written = vec![binding.clone()];
                drop(
                    peer.owner()
                        .spawn(move |server| server.peer_wrote(written, None)),
                );
            }
        }
    }
}

fn shared(binding: &BufferBinding) -> &PeerMemory {
    binding
        .peer
        .as_ref()
        .expect("a peer binding is built from shared memory")
}
