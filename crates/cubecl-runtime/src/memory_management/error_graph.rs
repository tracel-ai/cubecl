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

use crate::id::KernelId;
use crate::memory_management::ManagedMemoryId;
use crate::server::ServerError;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU64;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;

/// The id a tainted allocation carries, naming the failure that left its
/// bytes unwritten.
///
/// Opaque on purpose: a carrier gains a word and nothing else. It does not
/// learn what an error is, it cannot report one, and it has no opinion about
/// streams. `NonZero` so `Option<FailureId>` is that one word rather than two.
///
/// Wide because ids are never reused: one is minted for every write scope
/// that claims anything, which is every launch and every host copy, and a
/// narrower counter would run out. Reuse is what a 32-bit id would need, and
/// reuse is unsound here — [`prune`](ErrorGraph::prune) drops a node whose tag
/// count is zero, which a freshly minted node also has, so a recycled id lets
/// one scope's prune delete another's failure. Free in the carrier either way:
/// [`Tainted`](super::Taint) pads a `u32` out to eight bytes before its
/// ranges, which `a_failure_id_is_free_in_the_carrier` pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FailureId(NonZeroU64);

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
    /// zero and, being 64 bits wide, never wraps.
    minted: u64,
}

#[derive(Debug)]
struct Failure {
    error: ServerError,
    /// How many slices carry this id. The node lives while this is non-zero.
    tagged: u32,
    /// What this failure stopped, newest last, capped to the most recent —
    /// see [`Skipped`]. On the failure rather than on the buffers because a
    /// record stored on an allocation dies when that allocation does, and a
    /// chain of links would break as soon as an intermediate buffer is freed
    /// — the common case for a fused graph where only the last tensor is
    /// kept. Holds ids, never handles, so the record of what a failure
    /// stopped never retains the memory it names.
    skipped: Vec<Skipped>,
    /// Every skip, the capped list included: the report says how many are
    /// missing from the walk.
    skipped_total: u64,
}

/// One launch a failure stopped: the kernel that did not run, the buffer
/// whose claim stopped it, and what it would have produced.
///
/// A read of a downstream buffer walks these backwards — the record that
/// produced this buffer, then the record that produced what that one needed —
/// so the report names the path from the read back to the root instead of
/// only the root.
#[derive(Debug, Clone)]
pub struct Skipped {
    /// The kernel the skip stopped.
    pub kernel: KernelId,
    /// The claimed buffer the kernel would have read.
    pub needed: ManagedMemoryId,
    /// The buffers the kernel would have written, which now carry the same
    /// failure.
    pub produced: Vec<ManagedMemoryId>,
}

impl ErrorGraph {
    /// How many [`Skipped`] records one failure keeps. Newest win: the walk a
    /// read makes starts from the most recent buffers, so keeping the oldest
    /// would leave them with no entry to start from, while a deep chain
    /// reaching a gap before the root costs nothing — the root is on the node
    /// itself and never in the list.
    pub const MAX_SKIPPED: usize = 16;

    /// Hold `error` until nothing carries its id any more.
    ///
    /// The node starts carried by nothing, so a caller that taints no slice
    /// with it must hand it back through [`prune`](Self::prune) — otherwise
    /// the node waits forever for a decrement that is never coming.
    pub fn insert(&mut self, error: ServerError) -> FailureId {
        self.minted = self
            .minted
            .checked_add(1)
            .expect("a failure id was minted for every u64");
        let id = FailureId(NonZeroU64::new(self.minted).expect("minted starts above zero"));
        self.nodes.insert(
            id,
            Failure {
                error,
                tagged: 0,
                skipped: Vec::new(),
                skipped_total: 0,
            },
        );
        id
    }

    /// Record a launch `failure` stopped — see [`Skipped`]. Keeps the newest
    /// [`MAX_SKIPPED`](Self::MAX_SKIPPED) records and counts them all.
    pub fn skipped(&mut self, failure: FailureId, record: Skipped) {
        let Some(node) = self.nodes.get_mut(&failure) else {
            return;
        };
        node.skipped_total += 1;
        if node.skipped.len() == Self::MAX_SKIPPED {
            node.skipped.remove(0);
        }
        node.skipped.push(record);
    }

    /// The report a read of `memory` gets when `failure` claims its bytes:
    /// the root error, and the path from this buffer back toward it,
    /// reconstructed by walking the skip records backwards.
    pub fn report(&self, failure: FailureId, memory: ManagedMemoryId) -> Option<ServerError> {
        let node = self.nodes.get(&failure)?;

        let mut chain = Vec::new();
        let mut target = memory;
        let mut upper = node.skipped.len();
        while let Some(found) = node.skipped[..upper]
            .iter()
            .rposition(|record| record.produced.contains(&target))
        {
            let record = &node.skipped[found];
            chain.push(alloc::format!(
                "skipped `{}`: it needed memory {:?}, which carried the failure",
                record.kernel.short_name(),
                record.needed,
            ));
            target = record.needed;
            upper = found;
        }
        let dropped = node.skipped_total.saturating_sub(node.skipped.len() as u64);
        if !chain.is_empty() && dropped > 0 {
            chain.push(alloc::format!(
                "({dropped} older skip record(s) were dropped; the walk may stop before the root)"
            ));
        }

        Some(ServerError::Unwritten {
            failure: failure.0.get(),
            claimed: node.tagged,
            chain,
            root: Box::new(node.error.clone()),
            backtrace: BackTrace::capture(),
        })
    }

    /// The report a read owes for the claims it found: one error per distinct
    /// failure, however many of the buffers carry it.
    ///
    /// This is the shape of every "were these bytes written" answer in the
    /// system — [`FailureStore::ensure_written`](crate::stream::FailureStore::ensure_written)
    /// and any harness standing in for it — so the dedup and the wrapping live
    /// here rather than once per caller.
    ///
    /// # Errors
    ///
    /// [`ServerError::Several`] naming each failure once, in the order the
    /// claims were found. The caller has nothing to retry — the bytes are gone
    /// — so this is the answer to the read, not a hint to try again.
    pub fn reports(
        &self,
        claims: impl Iterator<Item = (FailureId, ManagedMemoryId)>,
    ) -> Result<(), ServerError> {
        let mut seen: Vec<FailureId> = Vec::new();
        let mut errors = Vec::new();

        for (failure, memory) in claims {
            if seen.contains(&failure) {
                continue;
            }
            seen.push(failure);
            // The full report: the root error, and the skip chain from this
            // buffer back toward it.
            if let Some(error) = self.report(failure, memory) {
                errors.push(error);
            }
        }

        match errors.is_empty() {
            true => Ok(()),
            false => Err(ServerError::Several {
                errors,
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// One more allocation carries `failure`.
    ///
    /// The other half of [`untag`](Self::untag), called only by the taint
    /// bookkeeping on the slices — [`Taint`](super::Taint) — which owns the
    /// invariant that a claim tags exactly once however many times it is
    /// re-tainted or split.
    pub(crate) fn tag(&mut self, failure: FailureId) {
        self.node_mut(failure).tagged += 1;
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
        // Saturating rather than `-= 1`: an unbalanced shed would wrap the
        // count in release and pin the node — and the error it holds — for the
        // life of the device, with every read of that allocation failing
        // forever. The floor drops the node instead, and the assertion makes
        // the shedding path that lost count loud in a test rather than silent
        // in production.
        debug_assert!(node.tagged > 0, "{failure} was shed more often than tagged");
        node.tagged = node.tagged.saturating_sub(1);
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

    /// How many failures the device is still holding — the bound the whole
    /// design rests on, which is why the property harness watches it.
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

        graph.tag(failure);
        graph.tag(failure);
        assert_eq!(graph.len(), 1);

        graph.untag(Some(failure));
        assert!(graph.error(failure).is_some(), "one carrier remains");

        graph.untag(Some(failure));
        assert!(
            graph.error(failure).is_none(),
            "nothing carries it, so it is gone"
        );
        assert!(graph.is_empty());
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

        graph.tag(failure);
        graph.replace(failure, error("launch"));

        match graph.error(failure) {
            Some(ServerError::Generic { reason, .. }) => assert_eq!(reason, "launch"),
            other => panic!("expected the replaced error, got {other:?}"),
        }

        graph.untag(Some(failure));
        assert!(graph.is_empty());
    }

    fn skip(
        kernel_name: KernelId,
        needed: ManagedMemoryId,
        produced: &[ManagedMemoryId],
    ) -> Skipped {
        Skipped {
            kernel: kernel_name,
            needed,
            produced: produced.to_vec(),
        }
    }

    fn memory_id(value: usize) -> ManagedMemoryId {
        ManagedMemoryId { value }
    }

    struct Fill;
    struct Matmul;
    struct Gelu;

    /// The walk a read makes: from the buffer asked about, backwards through
    /// the skip records, to the root — each hop the record that produced what
    /// the previous one needed.
    #[test]
    fn a_report_walks_the_skip_chain_back_to_the_root() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("fill_f32 failed to compile"));
        graph.tag(failure);

        let (root_out, mid, last) = (memory_id(77), memory_id(91), memory_id(103));
        graph.skipped(failure, skip(KernelId::new::<Matmul>(), root_out, &[mid]));
        graph.skipped(failure, skip(KernelId::new::<Gelu>(), mid, &[last]));

        let report = graph.report(failure, last).unwrap();
        let text = alloc::format!("{report}");
        let gelu = text.find("Gelu").expect("the newest hop comes first");
        let matmul = text.find("Matmul").expect("then the one it needed");
        assert!(gelu < matmul, "newest skip first, root last: {text}");
        assert!(
            text.contains("fill_f32 failed to compile"),
            "the root is always in the report: {text}"
        );
        assert!(
            text.contains(&alloc::format!("#{}", failure.0.get())),
            "the failure id ties reads of the same failure together: {text}"
        );

        // A buffer no record produced reports the root alone.
        let report = graph.report(failure, memory_id(555)).unwrap();
        let text = alloc::format!("{report}");
        assert!(!text.contains("Gelu") && text.contains("fill_f32 failed to compile"));
    }

    /// The cap keeps the newest records — the walk starts from the most
    /// recent buffers, so keeping the oldest would leave them with no entry
    /// to start from — and the report says what it dropped.
    #[test]
    fn the_skip_cap_keeps_the_newest_records() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("root"));
        graph.tag(failure);

        for i in 0..(ErrorGraph::MAX_SKIPPED + 4) {
            graph.skipped(
                failure,
                skip(KernelId::new::<Fill>(), memory_id(i), &[memory_id(i + 1)]),
            );
        }

        let newest = memory_id(ErrorGraph::MAX_SKIPPED + 4);
        let report = graph.report(failure, newest).unwrap();
        let text = alloc::format!("{report}");
        assert!(
            text.contains("Fill"),
            "the newest buffer still has an entry to walk from: {text}"
        );
        assert!(
            text.contains("4 older skip record(s) were dropped"),
            "and the report says the walk may stop early: {text}"
        );
    }

    /// Pruning is only for the untainted case: a failure something carries
    /// stays for its carriers.
    #[test]
    fn pruning_leaves_a_carried_failure_alone() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        graph.tag(failure);
        graph.prune(failure);

        assert!(graph.error(failure).is_some());
    }
}
