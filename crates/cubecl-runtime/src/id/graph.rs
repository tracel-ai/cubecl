use core::sync::atomic::{AtomicU64, Ordering};

/// Identifies a backend-owned captured graph.
///
/// [`end_capture`](crate::server::Server::end_capture) records a graph,
/// stores it in the backend's own registry, and returns this lightweight id;
/// [`replay`](crate::server::Server::replay) and
/// [`graph_destroy`](crate::server::Server::graph_destroy) take the id
/// back to look the graph up. Referencing the graph by id keeps the raw
/// executable inside the server — it never crosses the actor boundary in a box —
/// exactly as memory is referenced by [`Handle`](crate::server::Handle) rather
/// than by raw pointer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct GraphId {
    value: u64,
}

impl GraphId {
    /// An id no other graph of this process has.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        assert!(value != u64::MAX, "Graph ids should not run out");
        Self { value }
    }
}

impl Default for GraphId {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_graphs_never_share_an_id() {
        assert_ne!(GraphId::new(), GraphId::new());
    }
}
