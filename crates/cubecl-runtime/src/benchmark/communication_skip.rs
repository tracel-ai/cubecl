use crate::client::Client;
use alloc::vec::Vec;

/// Every collective over the given devices queues nothing while this lives, so nothing crosses a
/// link and its destinations keep whatever they held.
///
/// Results are wrong while it is held. It is for timing a run as if communication were free.
/// A [`ClientGroup`](crate::client::ClientGroup) skips when any member's device is held; a
/// per-device [`all_reduce`](Client::all_reduce) reads only its own, so hold every device it
/// spans.
pub struct CommunicationSkipGuard {
    clients: Vec<Client>,
}

impl CommunicationSkipGuard {
    /// Skips communication on the devices of `clients` until this drops.
    pub fn enter(clients: &[Client]) -> Self {
        for client in clients {
            client.communication_skips().enter();
        }
        Self {
            clients: clients.to_vec(),
        }
    }
}

impl Drop for CommunicationSkipGuard {
    fn drop(&mut self) {
        for client in &self.clients {
            client.communication_skips().leave();
        }
    }
}
