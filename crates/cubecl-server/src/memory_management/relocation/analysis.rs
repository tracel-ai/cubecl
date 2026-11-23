//! Which outdated pages a relocation empties, and in what order.

use crate::memory_management::ManagedMemoryBinding;
use crate::memory_management::memory_pool::MemoryPage;
use alloc::vec::Vec;

/// The outdated pages a relocation can empty: every page with a live
/// allocation on it, no [guard](crate::memory_management::PageGuard) keeping
/// it as it is, and no allocation bound.
///
/// An empty page needs no move, and a guarded one keeps what it holds. So does
/// a page with an allocation someone holds a binding of once every stream's
/// work is done: that binding is an address kept past the work, like a
/// read-back that points into the page on a runtime that reads in place.
/// Moving the allocation would leave it pointing at freed bytes.
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
    /// The allocation, bound rather than held: a handle of the relocation's own would count
    /// among its owners' while the move runs (see [`ManagedMemoryHandle::uncounted_binding`]).
    pub handle: ManagedMemoryBinding,
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
        if page.live().any(|slice| slice.handle.is_bound()) {
            return None;
        }
        let live: Vec<_> = page
            .live()
            .map(|slice| LiveAllocation {
                // Safety: the binding runs on the device thread in the stream's order, ahead of any
                // later launch of the buffer's owner.
                handle: unsafe { slice.handle.uncounted_binding() },
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
                    handle: crate::memory_management::ManagedMemoryHandle::new().binding(),
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
