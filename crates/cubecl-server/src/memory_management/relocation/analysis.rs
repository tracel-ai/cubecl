//! What the outdated pages hold, and which of them a relocation empties.

use crate::memory_management::ManagedMemoryHandle;
use crate::memory_management::memory_pool::MemoryPage;
use alloc::vec::Vec;

/// What a pool's outdated pages hold, page by page: the analysis a
/// relocation is planned from.
#[derive(Debug)]
pub struct OutdatedPages {
    pages: Vec<OutdatedPage>,
}

/// What an outdated page holds, for a relocation.
#[derive(Debug)]
pub enum OutdatedPage {
    /// Nothing live: a cleanup returns it without moving anything.
    Empty,
    /// Every live allocation on it can move: moving them all frees the page.
    Movable(MovablePage),
    /// A graph capture recorded an allocation on it, which keeps its address:
    /// the page stays whatever else moves, so nothing on it is worth moving.
    Captured,
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
    /// What each of `pages` holds.
    pub fn new<'a>(pages: impl IntoIterator<Item = &'a MemoryPage>) -> Self {
        Self {
            pages: pages.into_iter().map(OutdatedPage::new).collect(),
        }
    }

    /// The pages to empty, in the order to empty them.
    ///
    /// Every movable page, and only those: an empty page needs no move, and a
    /// captured one is held whatever moves. Cheapest first — the fewest live
    /// bytes — so that when the current pages run out of room for targets,
    /// the pages already emptied are as many as that room allowed.
    pub fn plan(self) -> Vec<MovablePage> {
        let mut movable: Vec<MovablePage> = self
            .pages
            .into_iter()
            .filter_map(|page| match page {
                OutdatedPage::Movable(page) => Some(page),
                OutdatedPage::Empty | OutdatedPage::Captured => None,
            })
            .collect();
        movable.sort_by_key(MovablePage::live_bytes);
        movable
    }
}

impl OutdatedPage {
    /// What `page` holds.
    pub fn new(page: &MemoryPage) -> Self {
        let mut live = page.live().peekable();
        if live.peek().is_none() {
            return OutdatedPage::Empty;
        }
        let mut allocations = Vec::new();
        for slice in live {
            if slice.captured {
                return OutdatedPage::Captured;
            }
            allocations.push(LiveAllocation {
                handle: slice.handle.clone(),
                size: slice.storage.size(),
            });
        }
        OutdatedPage::Movable(MovablePage { live: allocations })
    }
}

impl MovablePage {
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
    fn the_plan_skips_empty_and_captured_pages() {
        let plan = analysis([OutdatedPage::Empty, OutdatedPage::Captured, movable(&[1])]).plan();
        assert_eq!(sizes(&plan), [vec![1]]);
    }

    #[test]
    fn the_plan_empties_the_cheapest_pages_first() {
        let plan = analysis([movable(&[3, 2]), movable(&[1]), movable(&[4])]).plan();
        assert_eq!(sizes(&plan), [vec![1], vec![4], vec![3, 2]]);
    }

    fn analysis(pages: impl IntoIterator<Item = OutdatedPage>) -> OutdatedPages {
        OutdatedPages {
            pages: pages.into_iter().collect(),
        }
    }

    fn movable(sizes: &[u64]) -> OutdatedPage {
        OutdatedPage::Movable(MovablePage {
            live: sizes
                .iter()
                .map(|&size| LiveAllocation {
                    handle: ManagedMemoryHandle::new(),
                    size,
                })
                .collect(),
        })
    }

    fn sizes(plan: &[MovablePage]) -> Vec<Vec<u64>> {
        plan.iter()
            .map(|page| page.live.iter().map(|allocation| allocation.size).collect())
            .collect()
    }
}
