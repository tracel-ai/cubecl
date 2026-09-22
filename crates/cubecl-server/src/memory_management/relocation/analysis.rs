//! Which outdated pages a relocation empties, and in what order.

use crate::memory_management::ManagedMemoryHandle;
use crate::memory_management::memory_pool::MemoryPage;
use alloc::vec::Vec;

/// The outdated pages a relocation can empty: every page with a live
/// allocation on it and no [guard](crate::memory_management::PageGuard)
/// keeping it as it is.
///
/// An empty page needs no move, and a guarded one keeps what it holds.
#[derive(Debug)]
pub struct OutdatedPages {
    pages: Vec<MovablePage>,
}

/// An outdated page whose live allocations can all move.
#[derive(Debug)]
pub struct MovablePage {
    /// The allocations to move, as their owners hold them.
    pub live: Vec<LiveAllocation>,
}

/// One live allocation on an outdated page.
#[derive(Debug)]
pub struct LiveAllocation {
    /// The allocation, as its owners hold it.
    pub handle: ManagedMemoryHandle,
    /// Its size in bytes, which the move copies.
    pub size: u64,
}

impl OutdatedPages {
    /// The movable pages among `pages`.
    pub fn new<'a>(pages: impl IntoIterator<Item = &'a MemoryPage>) -> Self {
        Self {
            pages: pages
                .into_iter()
                .filter(|page| !page.is_guarded())
                .filter_map(MovablePage::new)
                .collect(),
        }
    }

    /// The pages to empty, cheapest first — the fewest live bytes — so that
    /// when the room for targets runs out, the pages already emptied are as
    /// many as that room allowed.
    pub fn plan(mut self) -> Vec<MovablePage> {
        self.pages.sort_by_key(MovablePage::live_bytes);
        self.pages
    }
}

impl MovablePage {
    /// What `page` holds, `None` when nothing live is on it.
    fn new(page: &MemoryPage) -> Option<Self> {
        let live: Vec<_> = page
            .live()
            .map(|slice| LiveAllocation {
                handle: slice.handle.clone(),
                size: slice.storage.size(),
            })
            .collect();
        (!live.is_empty()).then_some(Self { live })
    }

    /// The bytes moving this page's allocations copies.
    pub fn live_bytes(&self) -> u64 {
        self.live.iter().map(|allocation| allocation.size).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn the_plan_empties_the_cheapest_pages_first() {
        let plan = OutdatedPages {
            pages: vec![movable(&[3, 2]), movable(&[1]), movable(&[4])],
        }
        .plan();
        assert_eq!(sizes(&plan), [vec![1], vec![4], vec![3, 2]]);
    }

    fn movable(sizes: &[u64]) -> MovablePage {
        MovablePage {
            live: sizes
                .iter()
                .map(|&size| LiveAllocation {
                    handle: ManagedMemoryHandle::new(),
                    size,
                })
                .collect(),
        }
    }

    fn sizes(plan: &[MovablePage]) -> Vec<Vec<u64>> {
        plan.iter()
            .map(|page| page.live.iter().map(|allocation| allocation.size).collect())
            .collect()
    }
}
