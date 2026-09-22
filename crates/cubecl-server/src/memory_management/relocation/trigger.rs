//! When a relocation runs, and why.

use super::TargetRoom;

/// How close the device is to its capacity, for the next page of a given size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// The next page leaves room for at least another one.
    Low,
    /// The next page would leave less than another page of room: the last
    /// moment a relocation can still free what the device needs, since once
    /// the device refuses a page there is nowhere left to copy to.
    High,
    /// The device does not report its capacity, so nothing can be judged.
    Unknown,
}

impl MemoryPressure {
    /// The pressure on a device that holds `allocated` of its `capacity`
    /// bytes, for a next page of `page_size`.
    pub fn new(allocated: u64, capacity: Option<u64>, page_size: u64) -> Self {
        let Some(capacity) = capacity else {
            return MemoryPressure::Unknown;
        };
        match allocated + page_size > capacity.saturating_sub(page_size) {
            true => MemoryPressure::High,
            false => MemoryPressure::Low,
        }
    }
}

/// Why a relocation runs, which decides where it may put what it moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relocate {
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

impl Relocate {
    /// Where the relocation may reserve its targets: only in the room held
    /// when memory is what is short, anywhere otherwise.
    pub fn room(self) -> TargetRoom {
        match self {
            Relocate::Explicit | Relocate::MemoryPressure => TargetRoom::Held,
            Relocate::ArenaFull | Relocate::Capture => TargetRoom::MayAllocate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn pressure_is_high_once_the_next_page_leaves_less_than_another() {
        assert_eq!(
            MemoryPressure::new(10 * MIB, Some(24 * MIB), 4 * MIB),
            MemoryPressure::Low
        );
        assert_eq!(
            MemoryPressure::new(17 * MIB, Some(24 * MIB), 4 * MIB),
            MemoryPressure::High
        );
    }

    #[test]
    fn a_device_that_reports_no_capacity_is_not_judged() {
        assert_eq!(
            MemoryPressure::new(u64::MAX / 2, None, MIB),
            MemoryPressure::Unknown
        );
    }
}
