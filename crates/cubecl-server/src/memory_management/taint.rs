//! The tainted regions of one allocation.
//!
//! Whole-allocation taint answers whether a buffer can be trusted, but at the
//! wrong grain: a host write covering one row of a tensor would end the claim
//! on all of it, and a launch that failed writing one region would fail reads
//! of the rest. Both matter more once failures start changing control flow —
//! a partial write that clears a whole allocation un-skips every launch
//! downstream, which is garbage that carries no failure to report.
//!
//! So the claim is a set of byte ranges, each pointing at the failure that
//! made it. What a [`FailureId`] means still lives in the [`ErrorGraph`];
//! this type owns the carrier side of the refcount — one tag per failure per
//! allocation, held while that failure still claims at least one byte.
//!
//! # Why the storage is inline
//!
//! Every launch that writes anything runs the whole cycle on the success
//! path: the write scope claims each buffer on the way in and releases it on
//! the way out. So the case to price is not the clean slice, it is *one
//! failure over one range, held for the length of a launch* — and behind a
//! `Box<Vec<Vec<_>>>` that case cost three allocations and three frees per
//! written buffer per launch. Inline capacity for one claim of one range
//! makes it cost none, for about four words on a slice that already holds a
//! storage handle, a memory handle, a cursor and its padding. A second
//! failure, or a range a partial write split in two, spills to the heap as
//! before.

use super::{ErrorGraph, FailureId};
use core::ops::Range;
use smallvec::SmallVec;

/// The claims one allocation carries before spilling to the heap. One: a
/// launch's write scope claims the whole binding under a single failure, and
/// that is what every launch of a working program does.
type Claims = SmallVec<[FailureRanges; 1]>;

/// The ranges one claim covers before spilling. One: a claim starts as the
/// whole binding and only splits when a partial write lands inside it.
type Ranges = SmallVec<[Range<u64>; 1]>;

/// The tainted regions of one allocation, refcounted into the device's
/// [`ErrorGraph`].
///
/// # Assumptions
///
/// `failures`, in every function that takes it, is the [`ErrorGraph`] that
/// holds the failures this allocation's claims name.
///
/// # Guarantees
///
/// The allocation holds one tag on each failure that claims at least one of
/// its bytes, and none on any other.
#[derive(Debug, Default)]
pub struct Taint {
    recorder: ClaimRecorder,
}

/// Records the claims failures hold on the bytes of one allocation, tagging
/// each failure once while it claims a byte.
#[derive(Debug, Default)]
struct ClaimRecorder {
    /// One claim per failure, in the order the failures first claimed bytes
    /// (Arrival), found by their failure (Keyed) and scanned for the failure a
    /// range overlaps.
    entries: Claims,
}

/// One failure's claim on the allocation.
#[derive(Debug)]
struct FailureRanges {
    /// The failure that left the bytes unwritten.
    pub failure: FailureId,
    /// The bytes this failure left unwritten: disjoint, sorted, never empty
    /// (Ordered). More than one range because a partial write can split a
    /// claim in two.
    pub ranges: Ranges,
}

// A wider failure id costs the carrier nothing: the id is followed by padding
// out to the ranges' alignment either way. The id has to be wide, because ids
// are never reused and one is minted per write scope — and this is what makes
// that free. A carrier that grew would be a real cost on every slice in the
// pools, so the claim is pinned rather than asserted in a doc.
const _: () = {
    struct Narrow {
        _failure: core::num::NonZeroU32,
        _ranges: Ranges,
    }
    assert!(
        core::mem::size_of::<FailureRanges>() == core::mem::size_of::<Narrow>(),
        "a 64-bit failure id should fit in the padding a 32-bit one leaves"
    );
};

impl Taint {
    /// Point `range` at `failure`, releasing whatever claim other failures
    /// held on those bytes: the work that failed is their last writer now,
    /// and whatever the previous writer did or did not do stops mattering.
    ///
    /// Tainting is *set*, never *add*: re-tainting bytes this failure already
    /// claims changes nothing, so a loop failing the same way every iteration
    /// cannot grow the claim or pin the node harder. An empty range claims
    /// nothing.
    ///
    /// # Panics
    ///
    /// When `failure`, or a failure this allocation's claims name, is not
    /// one `failures` holds.
    pub fn taint(&mut self, range: Range<u64>, failure: FailureId, failures: &mut ErrorGraph) {
        self.recorder.record(range, failure, failures);
    }

    /// The bytes in `range` have a writer again: release every claim on them,
    /// and only on them — a write covering part of a buffer says nothing
    /// about the rest, which keeps carrying the failure that left it stale.
    ///
    /// # Panics
    ///
    /// When a failure this allocation's claims name is not one `failures`
    /// holds.
    pub fn written(&mut self, range: Range<u64>, failures: &mut ErrorGraph) {
        self.recorder.release(&range, None, failures);
    }

    /// The failure claiming any byte of `range`, if one does.
    ///
    /// A range overlapping several failures names one of them: the read fails
    /// either way, and the caller dedupes by id across a whole read anyway.
    pub fn failure(&self, range: &Range<u64>) -> Option<FailureId> {
        self.recorder.claiming(range)
    }

    /// Release every claim, for an allocation that stops existing — the slice
    /// is rebound, coalesced away, tombstoned or swept — since a tag must not
    /// outlive its carrier.
    ///
    /// # Panics
    ///
    /// When a failure this allocation's claims name is not one `failures`
    /// holds.
    pub fn clear(&mut self, failures: &mut ErrorGraph) {
        self.recorder.clear(failures);
    }

    /// Whether nothing claims any byte.
    pub fn is_clean(&self) -> bool {
        self.recorder.is_clean()
    }
}

impl ClaimRecorder {
    /// Point `range` at `failure`, as [`Taint::taint`] says.
    fn record(&mut self, range: Range<u64>, failure: FailureId, failures: &mut ErrorGraph) {
        if range.is_empty() {
            return;
        }
        self.release(&range, Some(failure), failures);
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.failure == failure)
        {
            Some(entry) => add(&mut entry.ranges, range),
            None => self.claim(range, failure, failures),
        }
    }

    /// Add a claim of `failure` on `range`, tagging it once.
    fn claim(&mut self, range: Range<u64>, failure: FailureId, failures: &mut ErrorGraph) {
        failures.tag(failure);
        self.entries.push(FailureRanges {
            failure,
            ranges: Ranges::from_buf([range]),
        });
    }

    /// Release every claim on `range` but `kept`'s, untagging each failure
    /// left claiming nothing.
    fn release(&mut self, range: &Range<u64>, kept: Option<FailureId>, failures: &mut ErrorGraph) {
        if range.is_empty() {
            return;
        }
        self.entries
            .retain_mut(|entry| Self::release_entry(entry, range, kept, failures));
    }

    /// Release `entry`'s claim on `range` unless it is `kept`'s, answering
    /// whether it still claims a byte.
    fn release_entry(
        entry: &mut FailureRanges,
        range: &Range<u64>,
        kept: Option<FailureId>,
        failures: &mut ErrorGraph,
    ) -> bool {
        if Some(entry.failure) == kept {
            return true;
        }
        subtract(&mut entry.ranges, range);
        Self::keep_claiming(entry, failures)
    }

    /// Whether `entry` still claims a byte, untagging its failure when it does not.
    fn keep_claiming(entry: &FailureRanges, failures: &mut ErrorGraph) -> bool {
        if !entry.ranges.is_empty() {
            return true;
        }
        failures.untag(Some(entry.failure));
        false
    }

    /// The failure claiming any byte of `range`.
    fn claiming(&self, range: &Range<u64>) -> Option<FailureId> {
        self.entries
            .iter()
            .find(|entry| entry.ranges.iter().any(|held| overlaps(held, range)))
            .map(|entry| entry.failure)
    }

    /// Release every claim.
    fn clear(&mut self, failures: &mut ErrorGraph) {
        for entry in self.entries.drain(..) {
            failures.untag(Some(entry.failure));
        }
    }

    /// Whether nothing claims any byte.
    fn is_clean(&self) -> bool {
        self.entries.is_empty()
    }
}

fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    // Their intersection is non-empty — which an empty range's never is, so a
    // zero-sized binding neither trips a claim nor loses one.
    a.start.max(b.start) < a.end.min(b.end)
}

/// Remove `cut` from `ranges`, splitting a range it lands inside.
fn subtract(ranges: &mut Ranges, cut: &Range<u64>) {
    let mut index = 0;
    while index < ranges.len() {
        index += subtract_at(ranges, index, cut);
    }
}

/// Remove `cut` from the range of `ranges` at `index`, answering how many
/// ranges stand where it stood.
fn subtract_at(ranges: &mut Ranges, index: usize, cut: &Range<u64>) -> usize {
    let held = ranges[index].clone();
    if !overlaps(&held, cut) {
        return 1;
    }
    let left = held.start..cut.start.min(held.end);
    let right = cut.end.max(held.start)..held.end;
    ranges.remove(index);
    let kept: SmallVec<[Range<u64>; 2]> = [left, right]
        .into_iter()
        .filter(|range| !range.is_empty())
        .collect();
    let count = kept.len();
    ranges.insert_many(index, kept);
    count
}

/// Add `new` to `ranges`, fusing whatever it overlaps or touches so the list
/// stays disjoint and sorted.
fn add(ranges: &mut Ranges, mut new: Range<u64>) {
    ranges.retain(|held| {
        // Touching counts: [0, 10) and [10, 20) fuse into [0, 20).
        let fuses = held.start <= new.end && new.start <= held.end;
        if fuses {
            new = new.start.min(held.start)..new.end.max(held.end);
        }
        !fuses
    });
    let at = ranges
        .iter()
        .position(|held| new.end < held.start)
        .unwrap_or(ranges.len());
    ranges.insert(at, new);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerError;
    use alloc::string::ToString;

    fn error(reason: &str) -> ServerError {
        ServerError::Generic {
            reason: reason.to_string(),
            backtrace: Default::default(),
        }
    }

    /// The precision this type exists for: a write covering part of a buffer
    /// releases the claim on those bytes and only those bytes.
    #[test]
    fn a_partial_write_releases_only_the_bytes_it_covers() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));

        taint.taint(0..100, failure, &mut graph);
        taint.written(40..60, &mut graph);

        assert_eq!(taint.failure(&(0..40)), Some(failure));
        assert_eq!(taint.failure(&(40..60)), None, "these bytes were written");
        assert_eq!(taint.failure(&(60..100)), Some(failure));
        assert!(!graph.is_empty(), "the split claim still pins the node");

        taint.written(0..40, &mut graph);
        taint.written(60..100, &mut graph);
        assert!(taint.is_clean());
        assert!(graph.is_empty(), "the last byte released the node");
    }

    /// Two failures claiming disjoint regions coexist, and a read of each
    /// region names its own.
    #[test]
    fn disjoint_claims_keep_their_own_failures() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let first = graph.insert(error("first"));
        let second = graph.insert(error("second"));

        taint.taint(0..50, first, &mut graph);
        taint.taint(50..100, second, &mut graph);

        assert_eq!(taint.failure(&(10..20)), Some(first));
        assert_eq!(taint.failure(&(60..70)), Some(second));
        assert_eq!(graph.len(), 2);

        taint.written(0..50, &mut graph);
        assert_eq!(graph.len(), 1, "first has no carrier left");
        assert_eq!(taint.failure(&(60..70)), Some(second));
    }

    /// A new failure claiming bytes an old one held takes them over — set,
    /// never add — and the old node goes exactly when its last byte does.
    #[test]
    fn a_new_failure_takes_the_bytes_it_claims() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let old = graph.insert(error("old"));
        let new = graph.insert(error("new"));

        taint.taint(0..100, old, &mut graph);
        taint.taint(25..75, new, &mut graph);

        assert_eq!(taint.failure(&(0..25)), Some(old));
        assert_eq!(taint.failure(&(30..40)), Some(new));
        assert_eq!(taint.failure(&(75..100)), Some(old));

        taint.taint(0..100, new, &mut graph);
        assert_eq!(graph.len(), 1, "old claims nothing any more");
        assert_eq!(taint.failure(&(0..100)), Some(new));
    }

    /// The loop trap, range-flavored: failing the same way on the same bytes
    /// every iteration keeps one entry, one tag, one node.
    #[test]
    fn retainting_the_same_bytes_counts_once() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));

        for _ in 0..3 {
            taint.taint(0..100, failure, &mut graph);
        }
        assert_eq!(graph.len(), 1);

        taint.written(0..100, &mut graph);
        assert!(graph.is_empty(), "one entry, one tag, one untag");
    }

    /// Adjacent claims of one failure fuse rather than accumulate, so a
    /// kernel failing tile by tile does not grow the list without bound.
    #[test]
    fn adjacent_claims_of_one_failure_fuse() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));

        taint.taint(0..10, failure, &mut graph);
        taint.taint(20..30, failure, &mut graph);
        taint.taint(10..20, failure, &mut graph);

        assert_eq!(taint.recorder.entries.len(), 1);
        assert_eq!(taint.recorder.entries[0].ranges.len(), 1);
        assert_eq!(taint.recorder.entries[0].ranges[0], 0..30);
    }

    /// Clearing releases every claim at once, for the slice that stops
    /// existing.
    #[test]
    fn clearing_releases_every_claim() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let first = graph.insert(error("first"));
        let second = graph.insert(error("second"));

        taint.taint(0..50, first, &mut graph);
        taint.taint(50..100, second, &mut graph);
        taint.clear(&mut graph);

        assert!(taint.is_clean());
        assert!(graph.is_empty());
    }

    /// An empty range claims nothing.
    #[test]
    fn an_empty_range_claims_nothing() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));

        taint.taint(10..10, failure, &mut graph);

        assert!(taint.is_clean());
    }

    /// An empty range trips nothing: a zero-sized binding has no bytes to
    /// distrust.
    #[test]
    fn an_empty_range_trips_nothing() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));

        taint.taint(0..100, failure, &mut graph);

        assert_eq!(taint.failure(&(50..50)), None);
    }

    /// An empty write releases nothing.
    #[test]
    fn an_empty_write_releases_nothing() {
        let mut graph = ErrorGraph::default();
        let mut taint = Taint::default();
        let failure = graph.insert(error("launch"));
        taint.taint(0..100, failure, &mut graph);

        taint.written(50..50, &mut graph);

        assert_eq!(taint.failure(&(0..100)), Some(failure));
    }
}
