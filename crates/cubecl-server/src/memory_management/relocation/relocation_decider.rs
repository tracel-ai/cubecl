//! When a relocation runs, and why.

use super::TargetRoom;

/// What a [`RelocationDecider`] reads of the pools a workload's allocations
/// are carved from, while a growth left some of them outdated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaState {
    /// The size the current pages are carved at, which the next page has.
    pub page_size_bytes: u64,
    /// Whether every slot holds a pool, so the next growth has nowhere to go.
    pub is_full: bool,
    /// What a relocation plan reads, to tell whether anything changed since
    /// the last plan found nothing to move.
    pub shape: ArenaShape,
}

/// What a relocation plan reads of the pools: how many pages the current one
/// has room on, and what the outdated ones hold and keep guarded.
///
/// Room freed inside the current pages does not show here, since slices free
/// as their owners drop them: that room is found by the plan after the next
/// page, growth or explicit cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaShape {
    /// Pages the current pool holds.
    pub current_page_count: u64,
    /// Pools a growth left behind.
    pub outdated_pool_count: usize,
    /// Pages those pools hold.
    pub outdated_page_count: u64,
    /// Of those, the ones a guard keeps where they are.
    pub outdated_guarded_count: usize,
}

/// Decides when a relocation is worth its copies: what the device holds, and
/// how the last relocation went.
#[derive(Debug, Clone, Default)]
pub struct RelocationDecider {
    /// The pools as the last relocation found them with nothing to move, so
    /// the next is not planned until something changed.
    stalled: Option<ArenaShape>,
}

/// Whether a relocation is wanted, and what stands in its way: plain data, so
/// the caller can [resolve](Self::resolve) it against the bytes every stream
/// holds without anything borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationNeed {
    /// A stream records a graph: the pages its recording touches keep their
    /// addresses until it seals, so nothing moves.
    Recording,
    /// Nothing is outdated, so a relocation has nothing to move.
    Nothing,
    /// Something is outdated, but no relocation is wanted now: the last one
    /// found nothing to move and nothing changed since, the device reports no
    /// capacity to judge by, or it still has room for another page.
    Settled,
    /// Wanted once the next page of `page_size_bytes` would leave a device of
    /// `capacity_bytes` less than another page of room.
    UnderPressure {
        /// What the device holds in total.
        capacity_bytes: u64,
        /// The size of the next page.
        page_size_bytes: u64,
    },
    /// A relocation is wanted now, for the reason it carries.
    Wanted(RelocationReason),
}

/// Why a relocation runs, which decides where it may put what it moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationReason {
    /// An explicit cleanup, which exists to give memory back.
    Explicit,
    /// The next page would leave the device too little room.
    MemoryPressure,
    /// Every page size the memory tracks is taken, so the next growth has
    /// nowhere to go, however much room the device has.
    ArenaFull,
    /// A graph capture is about to start: every page it touches is guarded
    /// for the graph's life, and a guarded page can no longer be emptied.
    Capture,
}

/// How close the device is to its capacity, for the next page of a given size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryPressure {
    /// The next page leaves room for at least another one.
    Low,
    /// The next page would leave less than another page of room: the last
    /// moment a relocation can still free what the device needs, since once
    /// the device refuses a page there is nowhere left to copy to.
    High,
}

impl RelocationDecider {
    /// Whether the outdated pools `arena` describes want a relocation, on a
    /// device that holds `capacity` bytes where it says.
    pub fn decide(&self, arena: &ArenaState, capacity: Option<u64>) -> RelocationNeed {
        if self.stalled == Some(arena.shape) {
            return RelocationNeed::Settled;
        }
        if arena.is_full {
            return RelocationNeed::Wanted(RelocationReason::ArenaFull);
        }
        capacity.map_or(RelocationNeed::Settled, |capacity_bytes| {
            RelocationNeed::UnderPressure {
                capacity_bytes,
                page_size_bytes: arena.page_size_bytes,
            }
        })
    }

    /// A relocation moved something: the next is planned whenever it is
    /// wanted.
    pub fn moved(&mut self) {
        self.stalled = None;
    }

    /// A relocation moved nothing: none is planned again until the pools
    /// `arena` describes change.
    pub fn stalled(&mut self, arena: &ArenaState) {
        self.stalled = Some(arena.shape);
    }
}

impl RelocationNeed {
    /// This need on a device whose memories hold `allocated` bytes, read only
    /// when the answer depends on them: pressure is resolved into a wanted
    /// relocation or a settled one.
    pub fn resolve(self, allocated: impl FnOnce() -> u64) -> RelocationNeed {
        match self {
            RelocationNeed::UnderPressure {
                capacity_bytes,
                page_size_bytes,
            } => MemoryPressure::new(allocated(), capacity_bytes, page_size_bytes).need(),
            need => need,
        }
    }

    /// This need, or [`Recording`](Self::Recording) when a stream is
    /// `recording` a graph.
    pub fn while_recording(self, recording: bool) -> RelocationNeed {
        if recording {
            return RelocationNeed::Recording;
        }
        self
    }

    /// The need of a device whose streams' memories need `needs`:
    /// [`Recording`](Self::Recording) when one records, [`Settled`](Self::Settled)
    /// when something is outdated in one of them, [`Nothing`](Self::Nothing)
    /// otherwise.
    pub fn of_device(needs: impl IntoIterator<Item = RelocationNeed>) -> RelocationNeed {
        needs
            .into_iter()
            .fold(RelocationNeed::Nothing, RelocationNeed::join)
    }

    /// Whether something is outdated for a relocation to move, wanted now or
    /// not.
    pub fn has_outdated(self) -> bool {
        !matches!(self, RelocationNeed::Recording | RelocationNeed::Nothing)
    }
}

impl RelocationNeed {
    /// The need of two memories of one device, as [`of_device`](Self::of_device) joins them.
    fn join(self, other: RelocationNeed) -> RelocationNeed {
        match (self, other) {
            (RelocationNeed::Recording, _) | (_, RelocationNeed::Recording) => {
                RelocationNeed::Recording
            }
            (RelocationNeed::Nothing, RelocationNeed::Nothing) => RelocationNeed::Nothing,
            _ => RelocationNeed::Settled,
        }
    }
}

impl RelocationReason {
    /// Where the relocation may reserve its targets: only in the room held
    /// when memory is what is short, anywhere otherwise.
    pub fn room(self) -> TargetRoom {
        match self {
            RelocationReason::Explicit | RelocationReason::MemoryPressure => TargetRoom::Held,
            RelocationReason::ArenaFull | RelocationReason::Capture => TargetRoom::MayAllocate,
        }
    }
}

impl MemoryPressure {
    /// The pressure on a device that holds `allocated` of its `capacity`
    /// bytes, for a next page of `page_size`.
    fn new(allocated: u64, capacity: u64, page_size: u64) -> Self {
        if allocated.saturating_add(page_size) > capacity.saturating_sub(page_size) {
            return MemoryPressure::High;
        }
        MemoryPressure::Low
    }

    /// What this pressure asks of a relocation.
    fn need(self) -> RelocationNeed {
        match self {
            MemoryPressure::High => RelocationNeed::Wanted(RelocationReason::MemoryPressure),
            MemoryPressure::Low => RelocationNeed::Settled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn arena(page_size_bytes: u64) -> ArenaState {
        ArenaState {
            page_size_bytes,
            is_full: false,
            shape: ArenaShape {
                current_page_count: 1,
                outdated_pool_count: 1,
                outdated_page_count: 1,
                outdated_guarded_count: 0,
            },
        }
    }

    #[test]
    fn pressure_is_high_once_the_next_page_leaves_less_than_another() {
        let need = RelocationDecider::default().decide(&arena(4 * MIB), Some(24 * MIB));
        assert_eq!(need.resolve(|| 10 * MIB), RelocationNeed::Settled);
        assert_eq!(
            need.resolve(|| 17 * MIB),
            RelocationNeed::Wanted(RelocationReason::MemoryPressure)
        );
    }

    #[test]
    fn only_pressure_reads_the_bytes() {
        for need in [
            RelocationNeed::Nothing,
            RelocationNeed::Settled,
            RelocationNeed::Recording,
            RelocationNeed::Wanted(RelocationReason::Capture),
        ] {
            assert_eq!(need.resolve(|| panic!("the bytes are never summed")), need);
        }
    }

    #[test]
    fn a_full_arena_relocates_whatever_the_device_holds() {
        let state = ArenaState {
            is_full: true,
            ..arena(4 * MIB)
        };
        assert_eq!(
            RelocationDecider::default().decide(&state, None),
            RelocationNeed::Wanted(RelocationReason::ArenaFull)
        );
    }

    #[test]
    fn a_device_that_reports_no_capacity_is_not_judged() {
        assert_eq!(
            RelocationDecider::default().decide(&arena(MIB), None),
            RelocationNeed::Settled
        );
    }

    #[test]
    fn a_relocation_that_moved_nothing_waits_for_the_pools_to_change() {
        let mut decider = RelocationDecider::default();
        let state = arena(4 * MIB);
        decider.stalled(&state);
        assert_eq!(
            decider.decide(&state, Some(24 * MIB)),
            RelocationNeed::Settled
        );

        let grown = ArenaState {
            shape: ArenaShape {
                current_page_count: 2,
                ..state.shape
            },
            ..state
        };
        assert_ne!(
            decider.decide(&grown, Some(24 * MIB)),
            RelocationNeed::Settled
        );
    }

    #[test]
    fn a_relocation_that_moved_something_is_planned_again() {
        let mut decider = RelocationDecider::default();
        let state = arena(4 * MIB);
        decider.stalled(&state);

        decider.moved();

        assert_ne!(
            decider.decide(&state, Some(24 * MIB)),
            RelocationNeed::Settled
        );
    }

    #[test]
    fn a_recording_stream_needs_nothing_moved() {
        assert_eq!(
            RelocationNeed::Settled.while_recording(true),
            RelocationNeed::Recording
        );
        assert_eq!(
            RelocationNeed::Settled.while_recording(false),
            RelocationNeed::Settled
        );
    }

    #[test]
    fn a_device_needs_what_its_streams_need_together() {
        use RelocationNeed::*;
        assert_eq!(RelocationNeed::of_device([Nothing, Nothing]), Nothing);
        assert_eq!(RelocationNeed::of_device([Nothing, Settled]), Settled);
        assert_eq!(
            RelocationNeed::of_device([Wanted(RelocationReason::ArenaFull), Nothing]),
            Settled
        );
        assert_eq!(RelocationNeed::of_device([Settled, Recording]), Recording);
    }

    #[test]
    fn only_a_need_with_something_outdated_has_it() {
        assert!(!RelocationNeed::Recording.has_outdated());
        assert!(!RelocationNeed::Nothing.has_outdated());
        assert!(RelocationNeed::Settled.has_outdated());
        assert!(RelocationNeed::Wanted(RelocationReason::Explicit).has_outdated());
    }

    #[test]
    fn memory_is_short_only_under_explicit_and_pressure_relocations() {
        assert_eq!(RelocationReason::Explicit.room(), TargetRoom::Held);
        assert_eq!(RelocationReason::MemoryPressure.room(), TargetRoom::Held);
        assert_eq!(RelocationReason::ArenaFull.room(), TargetRoom::MayAllocate);
        assert_eq!(RelocationReason::Capture.room(), TargetRoom::MayAllocate);
    }
}
