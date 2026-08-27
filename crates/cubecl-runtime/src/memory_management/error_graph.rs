//! The device-wide store of failures, refcounted by the allocations that
//! carry them.
//!
//! Whether a buffer can be trusted is a property of that memory, so the fact
//! lives on the allocation: a [`Slice`](super::memory_pool::Slice) carries the
//! [`FailureId`] of the failure that tainted it, or none. What an id *means*
//! lives here, device-wide, because a launch failing on one stream can taint a
//! slice owned by another and both have to point at the same thing.
//!
//! A node is dropped when nothing carries its id any more, which is a
//! reference count without atomics — the graph and the slices are both
//! reachable only under the device handle's mutex. So the graph prunes
//! itself, and what it prunes is exactly the failures nothing can still read:
//! the graph leaks if and only if the program leaks memory.

use crate::server::ServerError;
use core::num::NonZeroU32;
use cubecl_environment::collections::HashMap;

/// The id a tainted allocation carries, naming the failure that left its
/// bytes unwritten.
///
/// Opaque on purpose: a slice gains four bytes and nothing else. It does not
/// learn what an error is, it cannot report one, and it has no opinion about
/// streams. `NonZeroU32` so `Option<FailureId>` costs the slice a single word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FailureId(NonZeroU32);

impl core::fmt::Display for FailureId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Every failure the device is still holding, and how many allocations carry
/// each one.
#[derive(Debug, Default)]
pub struct ErrorGraph {
    nodes: HashMap<FailureId, Failure>,
    /// Ids handed out so far; the next one is `minted + 1`, which is never
    /// zero.
    minted: u32,
}

#[derive(Debug)]
struct Failure {
    error: ServerError,
    /// How many slices carry this id. The node lives while this is non-zero.
    tagged: u32,
}

impl ErrorGraph {
    /// Hold `error` until nothing carries its id any more.
    ///
    /// The node starts carried by nothing, so a caller that taints no slice
    /// with it must hand it back through [`prune`](Self::prune) — otherwise
    /// the node waits forever for a decrement that is never coming.
    pub fn insert(&mut self, error: ServerError) -> FailureId {
        self.minted = self
            .minted
            .checked_add(1)
            .expect("a failure id was minted for every u32");
        let id = FailureId(NonZeroU32::new(self.minted).expect("minted starts above zero"));
        self.nodes.insert(id, Failure { error, tagged: 0 });
        id
    }

    /// Point `slot` at `failure`, releasing whatever it held before.
    ///
    /// Tainting is *set*, never *add*: a slot already carrying `failure` is
    /// left alone, so a loop failing the same way every iteration cannot
    /// increment `tagged` against a single slice forever and pin the node for
    /// the life of the process.
    pub fn taint(&mut self, slot: &mut Option<FailureId>, failure: FailureId) {
        if *slot == Some(failure) {
            return;
        }
        let released = slot.replace(failure);
        self.node_mut(failure).tagged += 1;
        self.untag(released);
    }

    /// One fewer allocation carries `failure`; the node is dropped when none
    /// does.
    ///
    /// This is what every shedding path calls — a slice written again, rebound
    /// to a new allocation, coalesced away, tombstoned or swept — and the
    /// decrement is immediate rather than collected into a list drained
    /// later, so a node nothing can reach is never retained just because
    /// nothing got around to saying so.
    pub fn untag(&mut self, failure: Option<FailureId>) {
        let Some(failure) = failure else {
            return;
        };
        let node = self.node_mut(failure);
        node.tagged -= 1;
        if node.tagged == 0 {
            self.nodes.remove(&failure);
        }
    }

    /// Swap the error behind `failure` for `error`, leaving every carrier
    /// pointing at the new one.
    ///
    /// This is the exit half of a write scope: entry taints the write set
    /// with a provisional node, because the real failure does not exist yet,
    /// and exit lands the real one here. A missing node is left missing — the
    /// id outlived its carriers, so nothing can read the error either way.
    pub fn replace(&mut self, failure: FailureId, error: ServerError) {
        if let Some(node) = self.nodes.get_mut(&failure) {
            node.error = error;
        }
    }

    /// Drop `failure` if nothing took its id — for a failure that turned out
    /// to taint nothing, whose node would otherwise wait forever at zero.
    pub fn prune(&mut self, failure: FailureId) {
        if let Some(node) = self.nodes.get(&failure)
            && node.tagged == 0
        {
            self.nodes.remove(&failure);
        }
    }

    /// The error behind `failure`.
    ///
    /// `None` means the id outlived its node, which the refcount exists to
    /// prevent; a reader treats it as no failure rather than panicking on the
    /// device thread.
    pub fn error(&self, failure: FailureId) -> Option<&ServerError> {
        self.nodes.get(&failure).map(|node| &node.error)
    }

    /// How many failures the device is still holding.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the device is holding no failure at all.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn node_mut(&mut self, failure: FailureId) -> &mut Failure {
        self.nodes
            .get_mut(&failure)
            .expect("a carried failure id always has its node")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn error(reason: &str) -> ServerError {
        ServerError::Generic {
            reason: reason.to_string(),
            backtrace: Default::default(),
        }
    }

    /// The bound the whole design rests on: a node lives exactly as long as
    /// some slice carries its id.
    #[test]
    fn a_node_lives_while_something_carries_it_and_no_longer() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        let mut a = None;
        let mut b = None;
        graph.taint(&mut a, failure);
        graph.taint(&mut b, failure);
        assert_eq!(graph.len(), 1);

        graph.untag(a.take());
        assert!(graph.error(failure).is_some(), "b still carries it");

        graph.untag(b.take());
        assert!(
            graph.error(failure).is_none(),
            "nothing carries it, so it is gone"
        );
        assert!(graph.is_empty());
    }

    /// Tainting is set, never add: the most ordinary failing program — a loop
    /// failing the same way every iteration — must not pin a node forever.
    #[test]
    fn retainting_with_the_same_failure_counts_once() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        let mut slot = None;
        graph.taint(&mut slot, failure);
        graph.taint(&mut slot, failure);
        graph.taint(&mut slot, failure);

        graph.untag(slot.take());
        assert!(graph.is_empty(), "one slice, one tag, one untag");
    }

    /// A slice tainted again by a different failure releases the one it held,
    /// so the superseded node can go as soon as its last carrier moves on.
    #[test]
    fn a_new_failure_releases_the_one_the_slot_held() {
        let mut graph = ErrorGraph::default();
        let first = graph.insert(error("first"));
        let second = graph.insert(error("second"));

        let mut slot = None;
        graph.taint(&mut slot, first);
        graph.taint(&mut slot, second);

        assert!(graph.error(first).is_none(), "nothing carries it any more");
        assert!(graph.error(second).is_some());
        assert_eq!(slot, Some(second));
    }

    /// A failure that tainted nothing is pruned rather than retained: a dry
    /// run's compile error, say, has no buffer to name.
    #[test]
    fn a_failure_that_tainted_nothing_is_pruned() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("dry-run"));

        graph.prune(failure);
        assert!(graph.is_empty());
    }

    /// The exit half of a write scope: the provisional error a scope entered
    /// with is swapped for the real one, and every carrier follows.
    #[test]
    fn replacing_an_error_leaves_the_carriers_pointing_at_the_new_one() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("torn down"));

        let mut slot = None;
        graph.taint(&mut slot, failure);
        graph.replace(failure, error("launch"));

        match graph.error(failure) {
            Some(ServerError::Generic { reason, .. }) => assert_eq!(reason, "launch"),
            other => panic!("expected the replaced error, got {other:?}"),
        }

        graph.untag(slot.take());
        assert!(graph.is_empty());
    }

    /// Pruning is only for the untainted case: a failure something carries
    /// stays for its carriers.
    #[test]
    fn pruning_leaves_a_carried_failure_alone() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        let mut slot = None;
        graph.taint(&mut slot, failure);
        graph.prune(failure);

        assert!(graph.error(failure).is_some());
    }
}
