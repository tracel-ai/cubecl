//! When a relocation runs, and why.

use super::TargetRoom;

/// What a [`RelocationTrigger`] reads of the pools a workload's allocations
/// are carved from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaState {
    /// The size the current pages are carved at, which the next page has.
    pub page_size: u64,
    /// Whether any pool is outdated: something a relocation could empty.
    pub has_outdated: bool,
    /// Whether every slot holds a pool, so the next growth has nowhere to go.
    pub full: bool,
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
    pub current_pages: u64,
    /// Pools a growth left behind.
    pub outdated_pools: usize,
    /// Pages those pools hold.
    pub outdated_pages: u64,
    /// Of those, the ones a guard keeps where they are.
    pub outdated_guarded: usize,
}

/// Decides when a relocation is worth its copies: what the device holds, and
/// how the last relocation went.
#[derive(Debug, Clone)]
pub struct RelocationTrigger {
    /// What the device holds in total, where it says.
    capacity: Option<u64>,
    /// The pools as the last relocation found them with nothing to move, so
    /// the next is not planned until something changed.
    stalled: Option<ArenaShape>,
}

/// Whether a relocation is wanted, before the bytes the device holds are
/// known: plain data, so the caller can gather those bytes from every stream
/// without anything borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationNeed {
    /// Nothing is outdated, the last relocation found nothing to move and
    /// nothing changed since, or the device reports no capacity to judge by.
    Nothing,
    /// Every page size the pools track is taken.
    ArenaFull,
    /// Wanted once the next page of `page_size` would leave a device of
    /// `capacity` bytes less than another page of room.
    UnderPressure {
        /// What the device holds in total.
        capacity: u64,
        /// The size of the next page.
        page_size: u64,
    },
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

impl RelocationTrigger {
    /// A trigger for a device that holds `capacity` bytes, where it says.
    pub fn new(capacity: Option<u64>) -> Self {
        Self {
            capacity,
            stalled: None,
        }
    }

    /// Whether the pools in `arena` want a relocation.
    pub fn need(&self, arena: &ArenaState) -> RelocationNeed {
        if !arena.has_outdated || self.stalled == Some(arena.shape) {
            return RelocationNeed::Nothing;
        }
        if arena.full {
            return RelocationNeed::ArenaFull;
        }
        match self.capacity {
            Some(capacity) => RelocationNeed::UnderPressure {
                capacity,
                page_size: arena.page_size,
            },
            None => RelocationNeed::Nothing,
        }
    }

    /// Remember how a relocation went: one that moved nothing is not planned
    /// again until the pools in `arena` change.
    pub fn settled(&mut self, moved: bool, arena: &ArenaState) {
        self.stalled = (!moved).then_some(arena.shape);
    }
}

impl RelocationNeed {
    /// Why to relocate now, if at all, on a device whose memories hold
    /// `allocated` bytes. Only asked when the answer depends on it.
    pub fn reason(self, allocated: impl FnOnce() -> u64) -> Option<RelocationReason> {
        match self {
            RelocationNeed::Nothing => None,
            RelocationNeed::ArenaFull => Some(RelocationReason::ArenaFull),
            RelocationNeed::UnderPressure {
                capacity,
                page_size,
            } => match MemoryPressure::new(allocated(), capacity, page_size) {
                MemoryPressure::High => Some(RelocationReason::MemoryPressure),
                MemoryPressure::Low => None,
            },
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
        match allocated + page_size > capacity.saturating_sub(page_size) {
            true => MemoryPressure::High,
            false => MemoryPressure::Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn arena(page_size: u64) -> ArenaState {
        ArenaState {
            page_size,
            has_outdated: true,
            full: false,
            shape: ArenaShape {
                current_pages: 1,
                outdated_pools: 1,
                outdated_pages: 1,
                outdated_guarded: 0,
            },
        }
    }

    #[test]
    fn pressure_is_high_once_the_next_page_leaves_less_than_another() {
        let need = RelocationTrigger::new(Some(24 * MIB)).need(&arena(4 * MIB));
        assert_eq!(need.reason(|| 10 * MIB), None);
        assert_eq!(
            need.reason(|| 17 * MIB),
            Some(RelocationReason::MemoryPressure)
        );
    }

    #[test]
    fn nothing_outdated_needs_nothing_and_sums_nothing() {
        let state = ArenaState {
            has_outdated: false,
            ..arena(4 * MIB)
        };
        let need = RelocationTrigger::new(Some(MIB)).need(&state);
        assert_eq!(need, RelocationNeed::Nothing);
        assert_eq!(need.reason(|| panic!("the bytes are never summed")), None);
    }

    #[test]
    fn a_full_arena_relocates_whatever_the_device_holds() {
        let state = ArenaState {
            full: true,
            ..arena(4 * MIB)
        };
        assert_eq!(
            RelocationTrigger::new(None).need(&state).reason(|| 0),
            Some(RelocationReason::ArenaFull)
        );
    }

    #[test]
    fn a_device_that_reports_no_capacity_is_not_judged() {
        assert_eq!(
            RelocationTrigger::new(None).need(&arena(MIB)),
            RelocationNeed::Nothing
        );
    }

    #[test]
    fn a_relocation_that_moved_nothing_waits_for_the_pools_to_change() {
        let mut trigger = RelocationTrigger::new(Some(24 * MIB));
        let state = arena(4 * MIB);
        trigger.settled(false, &state);
        assert_eq!(trigger.need(&state), RelocationNeed::Nothing);

        let grown = ArenaState {
            shape: ArenaShape {
                current_pages: 2,
                ..state.shape
            },
            ..state
        };
        assert_ne!(trigger.need(&grown), RelocationNeed::Nothing);
    }
}
