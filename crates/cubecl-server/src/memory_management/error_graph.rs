//! The device-wide store of failures, refcounted by the allocations that
//! carry them.
//!
//! Whether a buffer can be trusted is a property of that memory, so the fact
//! lives on the allocation: a slice carries the [`FailureId`] of the failure
//! that tainted it, or none. What an id *means* lives here, device-wide,
//! because a launch failing on one stream can taint a slice owned by another
//! and both have to point at the same thing.
//!
//! A node is dropped when nothing carries its id any more, which is a
//! reference count without atomics — the graph and the slices are both
//! reachable only under the device handle's mutex. So the graph prunes
//! itself, and what it prunes is exactly the failures nothing can still read:
//! the graph leaks if and only if the program leaks memory.

use super::FailureId;
use crate::id::KernelId;
use crate::memory_management::{ManagedMemoryBinding, ManagedMemoryId, WeakMemoryBinding};
use crate::server::{IoError, ServerError};
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::num::NonZeroU64;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;

/// Why the bytes of one buffer cannot be trusted, as a read finds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// A failure claims bytes of the allocation: the work that was going to
    /// write them did not run.
    Failed(FailureId, ManagedMemoryId),
    /// No reservation ever gave the memory a location. That failure has no
    /// slice to be recorded on, so without this the buffer carries nothing,
    /// passes every check, and a launch over it does nothing.
    Unallocated(ManagedMemoryId),
}

/// Every failure the device is still holding, and how many allocations carry
/// each one.
///
/// # Assumptions
///
/// Every [`FailureId`] handed to a function was minted by this graph's
/// [`insert`](Self::insert).
///
/// # Guarantees
///
/// A failure is held from its insertion until a prune finds nothing carrying
/// it, or until the last tag on it is shed, whichever comes first. A failure
/// recorded for a buffer that was never allocated is held until the buffer
/// gets another such record, which replaces it, or until the first
/// [`fail_unallocated`](Self::fail_unallocated) after the buffer is gone.
#[derive(Debug, Default)]
pub struct ErrorGraph {
    recorder: FailureRecorder,
}

/// One launch a failure stopped: the kernel that did not run, the buffer
/// whose claim stopped it, and what it would have produced.
///
/// A read of a downstream buffer walks these backwards — the record that
/// produced this buffer, then the record that produced what that one needed —
/// so the report names the path from the read back to the root instead of
/// only the root.
#[derive(Debug, Clone)]
pub struct SkippedLaunch {
    /// The kernel the skip stopped.
    pub kernel: KernelId,
    /// The claimed buffer the kernel would have read.
    pub needed: ManagedMemoryId,
    /// The buffers the kernel would have written, which now carry the same
    /// failure, in the launch's order (Membership).
    pub produced: Vec<ManagedMemoryId>,
}

impl ErrorGraph {
    /// How many [`SkippedLaunch`] records one failure keeps. Newest win: the
    /// walk a read makes starts from the most recent buffers, so keeping the
    /// oldest would leave them with no entry to start from, while a deep chain
    /// reaching a gap before the root costs nothing — the root is on the node
    /// itself and never in the list.
    pub const MAX_SKIPPED_LAUNCH_COUNT: usize = 16;

    /// Hold `error` until nothing carries its id any more.
    ///
    /// The node starts carried by nothing, so a caller that taints no slice
    /// with it must hand it back through [`prune`](Self::prune) — otherwise
    /// the node waits forever for a decrement that is never coming.
    pub fn insert(&mut self, error: ServerError) -> FailureId {
        self.recorder.insert(error)
    }

    /// Record a launch `failure` stopped. Keeps the newest
    /// [`MAX_SKIPPED_LAUNCH_COUNT`](Self::MAX_SKIPPED_LAUNCH_COUNT) records
    /// and counts them all; a failure no longer held records nothing.
    pub fn skipped(&mut self, failure: FailureId, record: SkippedLaunch) {
        self.recorder.record_skip(failure, record);
    }

    /// The report a read of `memory` gets when `failure` claims its bytes:
    /// the root error, and the path from this buffer back toward it,
    /// reconstructed by walking the skip records backwards.
    ///
    /// # Assumptions
    ///
    /// `failure` is still held.
    ///
    /// # Panics
    ///
    /// When `failure` is no longer held.
    pub fn report(&self, failure: FailureId, memory: ManagedMemoryId) -> ServerError {
        self.recorder.report(failure, memory)
    }

    /// The report a read owes for the claims it found: one error per distinct
    /// failure, however many of the buffers carry it, and one per buffer that
    /// was never allocated. The second kind is not in the graph, since nothing
    /// could carry it, but it is the same answer to the same question.
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
    pub fn reports(&self, claims: impl Iterator<Item = Claim>) -> Result<(), ServerError> {
        self.recorder.reports(claims)
    }

    /// One fewer allocation carries `failure`; the node is dropped when none
    /// does.
    ///
    /// This is what every shedding path calls — a slice written again, rebound
    /// to a new allocation, coalesced away, tombstoned or swept — and the
    /// decrement is immediate rather than collected into a list drained
    /// later, so a node nothing can reach is never retained just because
    /// nothing got around to saying so.
    ///
    /// # Assumptions
    ///
    /// `failure`, when there is one, is still held, and is shed no more often
    /// than it was [tagged](Self::tag).
    ///
    /// # Panics
    ///
    /// When `failure` is not held, or is shed more often than it was tagged.
    pub fn untag(&mut self, failure: Option<FailureId>) {
        self.recorder.untag(failure);
    }

    /// Swap the error behind `failure` for `error`, leaving every carrier
    /// pointing at the new one.
    ///
    /// This is the exit half of a write scope: entry taints the write set
    /// with a provisional node, because the real failure does not exist yet,
    /// and exit lands the real one here. A failure no longer held is left
    /// missing: the id outlived its carriers, so nothing can read the error
    /// either way.
    pub fn replace(&mut self, failure: FailureId, error: ServerError) {
        self.recorder.replace(failure, error);
    }

    /// Record `error` as the reason `memory` was never allocated, so a read of that memory
    /// reports that error.
    pub fn fail_unallocated(&mut self, memory: &ManagedMemoryBinding, error: ServerError) {
        self.recorder.fail_unallocated(memory, error);
    }

    /// The failure recorded for `memory` by [`fail_unallocated`](Self::fail_unallocated), if any.
    pub fn unallocated_failure(&self, memory: ManagedMemoryId) -> Option<FailureId> {
        self.recorder.unallocated_failure(memory)
    }

    /// Drop `failure` if nothing took its id — for a failure that turned out
    /// to taint nothing, whose node would otherwise wait forever at zero.
    pub fn prune(&mut self, failure: FailureId) {
        self.recorder.prune(failure);
    }

    /// The error `failure` was inserted with, or the one that last replaced
    /// it.
    ///
    /// # Assumptions
    ///
    /// `failure` is still held.
    ///
    /// # Panics
    ///
    /// When `failure` is no longer held.
    pub fn error(&self, failure: FailureId) -> &ServerError {
        self.recorder.error(failure)
    }

    /// How many failures the device is still holding — the bound the whole
    /// design rests on, which is why the property harness watches it.
    pub fn len(&self) -> usize {
        self.recorder.len()
    }

    /// Whether the device is holding no failure at all.
    pub fn is_empty(&self) -> bool {
        self.recorder.is_empty()
    }
}

impl ErrorGraph {
    /// One more allocation carries `failure`.
    ///
    /// The other half of [`untag`](Self::untag), called only by the taint
    /// bookkeeping on the slices — [`Taint`](super::Taint) — which owns the
    /// invariant that a claim tags exactly once however many times it is
    /// re-tainted or split.
    ///
    /// # Assumptions
    ///
    /// `failure` is still held.
    ///
    /// # Panics
    ///
    /// When `failure` is not held.
    pub(crate) fn tag(&mut self, failure: FailureId) {
        self.recorder.tag(failure);
    }
}

/// Records the failures the device holds, the allocations carrying each, and
/// the launches each stopped.
#[derive(Debug, Default)]
struct FailureRecorder {
    /// Each held failure by its id (Keyed).
    nodes: HashMap<FailureId, Failure>,
    /// The failures of buffers no allocation ever gave a location, which have no slice to
    /// carry a failure id, by the buffer's id (Keyed), the whole map swept for buffers that
    /// are gone on every new record. Each holds one tag on its failure, released once the buffer is gone.
    unallocated: HashMap<ManagedMemoryId, (FailureId, WeakMemoryBinding)>,
    /// Ids handed out so far; the next one is `minted + 1`, which is never
    /// zero and, being 64 bits wide, never wraps.
    minted: u64,
}

/// One held failure.
#[derive(Debug)]
struct Failure {
    /// The error the failure reports.
    pub error: ServerError,
    /// How many slices carry this id. The node lives while this is non-zero.
    pub tag_count: u32,
    /// What this failure stopped, oldest first, capped to the most recent
    /// (Arrival, the oldest dropped first). On the failure rather than on the
    /// buffers because a record stored on an allocation dies when that
    /// allocation does, and a chain of links would break as soon as an
    /// intermediate buffer is freed — the common case for a fused graph where
    /// only the last tensor is kept. Holds ids, never handles, so the record
    /// of what a failure stopped never retains the memory it names.
    pub skipped: VecDeque<SkippedLaunch>,
    /// Every skip, the capped list included: the report says how many are
    /// missing from the walk.
    pub skipped_count: u64,
}

impl FailureRecorder {
    fn insert(&mut self, error: ServerError) -> FailureId {
        self.minted = self
            .minted
            .checked_add(1)
            .expect("fewer failures than a u64 counts should ever be minted");
        let id = FailureId::from_minted(
            NonZeroU64::new(self.minted).expect("a minted count should start above zero"),
        );
        self.nodes.insert(
            id,
            Failure {
                error,
                tag_count: 0,
                skipped: VecDeque::new(),
                skipped_count: 0,
            },
        );
        id
    }

    fn record_skip(&mut self, failure: FailureId, record: SkippedLaunch) {
        let Some(node) = self.nodes.get_mut(&failure) else {
            return;
        };
        node.skipped_count += 1;
        if node.skipped.len() == ErrorGraph::MAX_SKIPPED_LAUNCH_COUNT {
            node.skipped.pop_front();
        }
        node.skipped.push_back(record);
    }

    fn report(&self, failure: FailureId, memory: ManagedMemoryId) -> ServerError {
        let node = self.node(failure);
        ServerError::Unwritten {
            failure: failure.to_u64(),
            claimed: node.tag_count,
            chain: node.skip_chain(memory),
            root: Box::new(node.error.clone()),
            backtrace: BackTrace::capture(),
        }
    }

    fn reports(&self, claims: impl Iterator<Item = Claim>) -> Result<(), ServerError> {
        let mut found = Vec::new();
        for claim in claims {
            let report = self.claim_report(claim, &found);
            found.extend(report);
        }
        if found.is_empty() {
            return Ok(());
        }
        Err(ServerError::Several {
            errors: found.into_iter().map(|(_, error)| error).collect(),
            backtrace: BackTrace::capture(),
        })
    }

    /// The report `claim` owes, with the failure it names: `None` when one
    /// of `found` already names that failure.
    fn claim_report(
        &self,
        claim: Claim,
        found: &[(Option<FailureId>, ServerError)],
    ) -> Option<(Option<FailureId>, ServerError)> {
        let (failure, memory) = match self.claimed(claim) {
            Ok(claimed) => claimed,
            Err(unallocated) => return Some((None, unallocated)),
        };
        if found.iter().any(|(named, _)| *named == Some(failure)) {
            return None;
        }
        Some((Some(failure), self.report(failure, memory)))
    }

    /// The failure `claim` names and the memory it claims, or the error of a
    /// buffer never allocated with no failure recorded for it.
    fn claimed(&self, claim: Claim) -> Result<(FailureId, ManagedMemoryId), ServerError> {
        match claim {
            Claim::Failed(failure, memory) => Ok((failure, memory)),
            Claim::Unallocated(memory) => self
                .unallocated
                .get(&memory)
                .map(|(failure, _)| (*failure, memory))
                .ok_or_else(|| unallocated(memory)),
        }
    }

    fn tag(&mut self, failure: FailureId) {
        self.node_mut(failure).tag_count += 1;
    }

    fn untag(&mut self, failure: Option<FailureId>) {
        let Some(failure) = failure else {
            return;
        };
        Self::untag_node(&mut self.nodes, failure);
    }

    fn replace(&mut self, failure: FailureId, error: ServerError) {
        if let Some(node) = self.nodes.get_mut(&failure) {
            node.error = error;
        }
    }

    fn fail_unallocated(&mut self, memory: &ManagedMemoryBinding, error: ServerError) {
        // Released lazily: a record whose buffer is gone is found here, when the next one is
        // made, rather than through a drop hook the buffer does not have.
        let nodes = &mut self.nodes;
        self.unallocated
            .retain(|_, (failure, binding)| Self::keep_alive(binding, *failure, nodes));

        let failure = self.insert(error);
        self.tag(failure);
        let previous = self
            .unallocated
            .insert(memory.id(), (failure, memory.downgrade()));
        self.untag(previous.map(|(previous, _)| previous));
    }

    fn unallocated_failure(&self, memory: ManagedMemoryId) -> Option<FailureId> {
        self.unallocated.get(&memory).map(|(failure, _)| *failure)
    }

    fn prune(&mut self, failure: FailureId) {
        if self
            .nodes
            .get(&failure)
            .is_some_and(|node| node.tag_count == 0)
        {
            self.nodes.remove(&failure);
        }
    }

    fn error(&self, failure: FailureId) -> &ServerError {
        &self.node(failure).error
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn node(&self, failure: FailureId) -> &Failure {
        self.nodes
            .get(&failure)
            .expect("a failure asked about should be held")
    }

    fn node_mut(&mut self, failure: FailureId) -> &mut Failure {
        self.nodes
            .get_mut(&failure)
            .expect("a carried failure id should always have its node")
    }
}

/// The cause is the reservation's own error, which the device thread logged
/// when it happened: nothing could carry it here.
fn unallocated(memory: ManagedMemoryId) -> ServerError {
    IoError::NotFound {
        backtrace: BackTrace::capture(),
        reason: alloc::format!(
            "memory {memory:?} was never allocated: the reservation behind it failed"
        )
        .into(),
    }
    .into()
}

impl FailureRecorder {
    /// Whether the unallocated buffer `binding` names is still alive, releasing
    /// the tag it held on `failure` among `nodes` once it is gone.
    fn keep_alive(
        binding: &WeakMemoryBinding,
        failure: FailureId,
        nodes: &mut HashMap<FailureId, Failure>,
    ) -> bool {
        if binding.is_alive() {
            return true;
        }
        Self::untag_node(nodes, failure);
        false
    }

    /// One fewer allocation carries `failure` among `nodes`; its node is dropped
    /// when none does.
    fn untag_node(nodes: &mut HashMap<FailureId, Failure>, failure: FailureId) {
        let node = nodes
            .get_mut(&failure)
            .expect("a carried failure id should always have its node");
        // A panic rather than a floor: an unbalanced shed would otherwise drop
        // the node while another carrier still names it, and that carrier's
        // bytes would read clean.
        node.tag_count = node
            .tag_count
            .checked_sub(1)
            .unwrap_or_else(|| panic!("{failure} should be shed no more often than it was tagged"));
        if node.tag_count == 0 {
            nodes.remove(&failure);
        }
    }
}

impl Failure {
    /// The path from `memory` back toward the root, walking its skip
    /// records backwards, and a note of the records the cap dropped when the walk
    /// may stop before the root.
    fn skip_chain(&self, memory: ManagedMemoryId) -> Vec<alloc::string::String> {
        let mut chain = Vec::new();
        let mut target = memory;
        let mut upper = self.skipped.len();
        while let Some(found) = self
            .skipped
            .range(..upper)
            .rposition(|record| record.produced.contains(&target))
        {
            let record = &self.skipped[found];
            chain.push(alloc::format!(
                "skipped `{}`: it needed memory {:?}, which carried the failure",
                record.kernel.short_name(),
                record.needed,
            ));
            target = record.needed;
            upper = found;
        }
        let dropped = self.skipped_count - self.skipped.len() as u64;
        if !chain.is_empty() && dropped > 0 {
            chain.push(alloc::format!(
                "({dropped} older skip record(s) were dropped; the walk may stop before the root)"
            ));
        }
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;
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
        assert_eq!(graph.len(), 1, "one carrier remains");

        graph.untag(Some(failure));
        assert!(graph.is_empty(), "nothing carries it, so it is gone");
    }

    /// Untagging nothing changes nothing.
    #[test]
    fn untagging_nothing_changes_nothing() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));
        graph.tag(failure);

        graph.untag(None);

        assert_eq!(graph.len(), 1);
    }

    /// A failure shed more often than it was tagged is a broken count, which
    /// stops rather than reading a carried failure as clean.
    #[test]
    #[should_panic(expected = "a carried failure id should always have its node")]
    fn an_unbalanced_shed_panics() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));
        graph.tag(failure);
        graph.untag(Some(failure));

        graph.untag(Some(failure));
    }

    /// The error of a failure no longer held is a broken assumption.
    #[test]
    #[should_panic(expected = "a failure asked about should be held")]
    fn the_error_of_a_dropped_failure_panics() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("dry-run"));
        graph.prune(failure);

        graph.error(failure);
    }

    /// A failure that tainted nothing is pruned rather than retained: a
    /// discarded launch's compile error, say, has no buffer to name.
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
            ServerError::Generic { reason, .. } => assert_eq!(reason, "launch"),
            other => panic!("expected the replaced error, got {other:?}"),
        }
    }

    /// Replacing the error of a failure no longer held leaves it missing.
    #[test]
    fn replacing_a_dropped_failure_leaves_it_missing() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("dry-run"));
        graph.prune(failure);

        graph.replace(failure, error("launch"));

        assert!(graph.is_empty());
    }

    fn skip(
        kernel_name: KernelId,
        needed: ManagedMemoryId,
        produced: &[ManagedMemoryId],
    ) -> SkippedLaunch {
        SkippedLaunch {
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

        let report = graph.report(failure, last);
        let text = alloc::format!("{report}");
        let gelu = text.find("Gelu").expect("the newest hop should come first");
        let matmul = text.find("Matmul").expect("then the one it needed should");
        assert!(gelu < matmul, "newest skip first, root last: {text}");
        assert!(
            text.contains("fill_f32 failed to compile"),
            "the root is always in the report: {text}"
        );
        assert!(
            text.contains(&alloc::format!("#{}", failure.to_u64())),
            "the failure id ties reads of the same failure together: {text}"
        );
    }

    /// A buffer no record produced reports the root alone.
    #[test]
    fn a_buffer_no_skip_produced_reports_the_root_alone() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("fill_f32 failed to compile"));
        graph.tag(failure);
        graph.skipped(
            failure,
            skip(KernelId::new::<Gelu>(), memory_id(1), &[memory_id(2)]),
        );

        let report = graph.report(failure, memory_id(555));

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

        for i in 0..(ErrorGraph::MAX_SKIPPED_LAUNCH_COUNT + 4) {
            graph.skipped(
                failure,
                skip(KernelId::new::<Fill>(), memory_id(i), &[memory_id(i + 1)]),
            );
        }

        let newest = memory_id(ErrorGraph::MAX_SKIPPED_LAUNCH_COUNT + 4);
        let report = graph.report(failure, newest);
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

    /// A skip of a failure no longer held records nothing.
    #[test]
    fn a_skip_of_a_dropped_failure_records_nothing() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("dry-run"));
        graph.prune(failure);

        graph.skipped(failure, skip(KernelId::new::<Fill>(), memory_id(1), &[]));

        assert!(graph.is_empty());
    }

    /// Pruning is only for the untainted case: a failure something carries
    /// stays for its carriers.
    #[test]
    fn pruning_leaves_a_carried_failure_alone() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        graph.tag(failure);
        graph.prune(failure);

        assert_eq!(graph.len(), 1);
    }

    /// A read reports each failure once, however many buffers carry it.
    #[test]
    fn a_read_reports_each_failure_once() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));
        graph.tag(failure);

        let reported = graph
            .reports(
                [
                    Claim::Failed(failure, memory_id(1)),
                    Claim::Failed(failure, memory_id(2)),
                ]
                .into_iter(),
            )
            .expect_err("the buffers should carry a failure");

        match reported {
            ServerError::Several { errors, .. } => assert_eq!(errors.len(), 1),
            other => panic!("expected one report per failure, got {other:?}"),
        }
    }

    /// A read with no claim is clean.
    #[test]
    fn a_read_with_no_claim_is_clean() {
        let graph = ErrorGraph::default();

        assert!(graph.reports(core::iter::empty()).is_ok());
    }

    /// A buffer that never got an allocation reports the failure recorded for it.
    #[test]
    fn an_unallocated_failure_is_reported() {
        let mut graph = ErrorGraph::default();
        let memory = ManagedMemoryHandle::new();

        graph.fail_unallocated(
            &memory.to_binding(),
            crate::driver::DevicePoison::new("the device is poisoned").into(),
        );
        let reported = graph
            .reports([Claim::Unallocated(memory.descriptor().id())].into_iter())
            .expect_err("the buffer should carry a failure");

        assert!(reported.is_device_poisoned(), "{reported}");
    }

    /// A buffer never allocated, with nothing recorded for it, never reads
    /// clean.
    #[test]
    fn an_unrecorded_unallocated_buffer_never_reads_clean() {
        let graph = ErrorGraph::default();
        let other = ManagedMemoryHandle::new();

        let unrecorded = graph
            .reports([Claim::Unallocated(other.descriptor().id())].into_iter())
            .expect_err("an unallocated buffer should never read clean");

        assert!(
            unrecorded.to_string().contains("never allocated"),
            "{unrecorded}"
        );
    }

    /// The record of an unallocated buffer goes once the buffer does, when
    /// the next record is made, and a second record of one buffer replaces
    /// its first.
    #[test]
    fn an_unallocated_failure_is_released_with_its_buffer() {
        let mut graph = ErrorGraph::default();
        let memory = ManagedMemoryHandle::new();
        let other = ManagedMemoryHandle::new();
        graph.fail_unallocated(&memory.to_binding(), error("first"));
        graph.fail_unallocated(&other.to_binding(), error("again"));
        graph.fail_unallocated(&other.to_binding(), error("again"));
        assert_eq!(
            graph.len(),
            2,
            "the second record of `other` replaced its first"
        );

        drop(memory);
        graph.fail_unallocated(&other.to_binding(), error("again"));

        assert_eq!(graph.len(), 1, "the dropped buffer's failure is released");
    }

    /// Shedding a failure never tagged is a broken count, which stops rather
    /// than reading a carried failure as clean.
    #[test]
    #[should_panic(expected = "should be shed no more often than it was tagged")]
    fn shedding_an_untagged_failure_panics() {
        let mut graph = ErrorGraph::default();
        let failure = graph.insert(error("launch"));

        graph.untag(Some(failure));
    }
}
