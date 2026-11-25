//! Which outdated pages a relocation empties, and in what order.

use crate::memory_management::ManagedMemoryBinding;
use crate::memory_management::memory_pool::PageSuballocator;
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
    pub(crate) fn new<'a>(pages: impl IntoIterator<Item = &'a PageSuballocator>) -> Self {
        let mut ordered: Vec<MovablePage> = Vec::new();
        for page in pages
            .into_iter()
            .filter(|page| !page.view().is_guarded)
            .filter_map(MovablePage::new)
        {
            // After every page that costs no more.
            let at = ordered.partition_point(|held| held.live_bytes() <= page.live_bytes());
            ordered.insert(at, page);
        }
        Self { pages: ordered }
    }

    /// The pages to empty, cheapest first — the fewest live bytes — so that
    /// when the room for targets runs out, the pages already emptied are as
    /// many as that room allowed.
    pub(crate) fn into_plan(self) -> Vec<MovablePage> {
        self.pages
    }
}

impl MovablePage {
    /// What `page` holds, `None` when nothing live is on it or one of its
    /// allocations is bound.
    fn new(page: &PageSuballocator) -> Option<Self> {
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
        MemoryLocation,
        memory_pool::{PageMapping, PageSuballocator},
    };
    use crate::storage::{StorageHandle, StorageId, StorageUtilization};
    use alloc::vec;

    /// The plan empties the pages with the fewest live bytes first, pages of
    /// equal cost in the order found.
    #[test]
    fn the_plan_empties_the_cheapest_pages_first() {
        let pages = [&[3, 2][..], &[1], &[4], &[1]].map(|sizes| {
            let mut page = page();
            let held: Vec<_> = sizes
                .iter()
                .map(|size| {
                    page.suballocate_bytes(*size)
                        .expect("the page should have room")
                })
                .collect();
            (page, held)
        });

        let plan = OutdatedPages::new(pages.iter().map(|(page, _)| page)).into_plan();

        assert_eq!(sizes(&plan), [vec![1], vec![1], vec![4], vec![3, 2]]);
    }

    /// A page with nothing live is not movable.
    #[test]
    fn an_empty_page_does_not_move() {
        let mut empty = page();
        core::mem::drop(empty.suballocate_bytes(64));

        assert!(OutdatedPages::new([&empty]).into_plan().is_empty());
    }

    /// A guarded page is not movable.
    #[test]
    fn a_guarded_page_does_not_move() {
        let mut guarded = page();
        let _slice = guarded.suballocate_bytes(64);
        let _guard = guarded.guard();

        assert!(OutdatedPages::new([&guarded]).into_plan().is_empty());
    }

    /// A page holding a bound allocation is not movable.
    #[test]
    fn a_page_with_a_bound_allocation_does_not_move() {
        let mut bound = page();
        let slice = bound
            .suballocate_bytes(64)
            .expect("the page should have room");
        let _binding = slice.clone().binding();

        assert!(OutdatedPages::new([&bound]).into_plan().is_empty());
    }

    /// A page of unbound live allocations moves them all.
    #[test]
    fn a_page_of_unbound_live_allocations_moves() {
        let mut movable = page();
        let _slice = movable.suballocate_bytes(64);

        assert_eq!(
            sizes(&OutdatedPages::new([&movable]).into_plan()),
            [vec![64]]
        );
    }

    fn page() -> PageSuballocator {
        PageSuballocator::new(
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
