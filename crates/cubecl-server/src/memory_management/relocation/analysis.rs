//! Which outdated pages a relocation empties, and in what order.

use crate::memory_management::ManagedMemoryBinding;
use crate::memory_management::memory_pool::PageCarver;
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
pub(crate) struct OutdatedPages {
    /// The movable pages, cheapest first: the fewest live bytes (Ordered),
    /// kept in that order as they are found.
    pages: Vec<MovablePage>,
}

/// An outdated page whose live allocations can all move.
#[derive(Debug)]
pub(crate) struct MovablePage {
    /// The allocations to move, as their owners hold them, in the page's
    /// offset order (Arrival).
    pub live: Vec<LiveAllocation>,
}

/// One live allocation on an outdated page.
#[derive(Debug)]
pub(crate) struct LiveAllocation {
    /// The allocation, bound rather than held: a handle of the relocation's own would count
    /// among its owners' while the move runs (see [`ManagedMemoryHandle::to_binding`](crate::memory_management::ManagedMemoryHandle::to_binding)).
    pub handle: ManagedMemoryBinding,
    /// The bytes the move copies.
    pub size_bytes: u64,
}

impl OutdatedPages {
    /// The movable pages among `pages`, cheapest first, pages of equal cost
    /// in the order found.
    pub(crate) fn new<'a>(pages: impl IntoIterator<Item = &'a PageCarver>) -> Self {
        let mut ordered = Self { pages: Vec::new() };
        pages
            .into_iter()
            .filter(|page| !page.view().is_guarded)
            .filter_map(MovablePage::new)
            .for_each(|page| ordered.insert(page));
        ordered
    }

    /// The pages to empty, cheapest first — the fewest live bytes — so that
    /// when the room for targets runs out, the pages already emptied are as
    /// many as that room allowed.
    pub(crate) fn into_plan(self) -> Vec<MovablePage> {
        self.pages
    }

    /// Place `page` after every page that costs no more.
    fn insert(&mut self, page: MovablePage) {
        let at = self
            .pages
            .partition_point(|held| held.live_bytes() <= page.live_bytes());
        self.pages.insert(at, page);
    }
}

impl MovablePage {
    /// What `page` holds, `None` when nothing live is on it or one of its
    /// allocations is bound.
    fn new(page: &PageCarver) -> Option<Self> {
        if page.view().live().any(|slice| slice.handle.is_bound()) {
            return None;
        }
        let live: Vec<_> = page
            .view()
            .live()
            .map(|slice| LiveAllocation {
                handle: slice.handle.to_binding(),
                size_bytes: slice.storage.size(),
            })
            .collect();
        (!live.is_empty()).then_some(Self { live })
    }

    /// The bytes moving this page's allocations copies.
    pub(crate) fn live_bytes(&self) -> u64 {
        self.live
            .iter()
            .map(|allocation| allocation.size_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::{
        ManagedMemoryHandle, MemoryLocation,
        memory_pool::{PageCarver, PageMapping},
    };
    use crate::storage::{StorageHandle, StorageId, StorageUtilization};
    use alloc::vec;

    #[test]
    fn the_plan_empties_the_cheapest_pages_first() {
        let mut pages = OutdatedPages { pages: Vec::new() };
        for page in [movable(&[3, 2]), movable(&[1]), movable(&[4])] {
            pages.insert(page);
        }
        let plan = pages.into_plan();
        assert_eq!(sizes(&plan), [vec![1], vec![4], vec![3, 2]]);
    }

    /// A page with nothing live, a guarded one and one holding a bound
    /// allocation are not movable; one with live allocations is.
    #[test]
    fn only_unguarded_pages_with_unbound_live_allocations_move() {
        let mut empty = page();
        let mut guarded = page();
        let mut bound = page();
        let mut movable = page();
        core::mem::drop(empty.carve_bytes(64));
        let _guarded_slice = guarded.carve_bytes(64);
        let _guard = guarded.guard();
        let bound_slice = bound.carve_bytes(64).expect("the page should have room");
        let _binding = bound_slice.clone().binding();
        let _movable_slice = movable.carve_bytes(64);

        let plan = OutdatedPages::new([&empty, &guarded, &bound, &movable]).into_plan();

        assert_eq!(sizes(&plan), [vec![64]]);
    }

    fn page() -> PageCarver {
        PageCarver::new(
            StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: 1024,
                },
            ),
            4,
            MemoryLocation::new(0, 0, 0),
            PageMapping::Eager,
        )
    }

    fn movable(sizes: &[u64]) -> MovablePage {
        MovablePage {
            live: sizes
                .iter()
                .map(|&size_bytes| LiveAllocation {
                    handle: ManagedMemoryHandle::new().binding(),
                    size_bytes,
                })
                .collect(),
        }
    }

    fn sizes(plan: &[MovablePage]) -> Vec<Vec<u64>> {
        plan.iter()
            .map(|page| {
                page.live
                    .iter()
                    .map(|allocation| allocation.size_bytes)
                    .collect()
            })
            .collect()
    }
}
