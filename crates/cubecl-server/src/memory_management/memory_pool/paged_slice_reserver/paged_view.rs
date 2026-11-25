use super::PagedSliceReserver;
use crate::memory_management::{
    MemoryPoolKind, MemoryPoolReport,
    memory_pool::{PageSuballocator, sum_usage},
};

/// What a [`PagedSliceReserver`] holds, as its readers see it: a handle on
/// the pool that reads each part only when asked.
#[derive(Clone, Copy)]
pub struct PagedView<'a> {
    pool: &'a PagedSliceReserver,
}

impl<'a> PagedView<'a> {
    /// A view of `pool`.
    pub fn new(pool: &'a PagedSliceReserver) -> Self {
        Self { pool }
    }

    /// The pages the pool holds, in the order a location numbers them.
    pub fn pages(&self) -> &'a [PageSuballocator] {
        &self.pool.pages
    }

    /// Whether graphs guard every page the pool holds, and it holds some: room
    /// for work outside a capture comes only from a new page then.
    pub fn is_pinned(&self) -> bool {
        !self.pool.pages.is_empty() && self.pool.pages.iter().all(|page| page.view().is_guarded)
    }

    /// The pool's shape, usage and high-water marks.
    pub fn report(&self) -> MemoryPoolReport {
        let pool = self.pool;
        MemoryPoolReport {
            kind: MemoryPoolKind::Sliced {
                page_size: pool.layout.page_size_bytes,
                max_slice_size: pool.layout.max_slice_bytes,
            },
            usage: sum_usage(pool.pages.iter().map(|page| page.view().usage())),
            page_count: pool.pages.len() as u64,
            peak_page_count: pool.peak_pages,
            unmapped_page_count: pool
                .pages
                .iter()
                .filter(|page| !page.view().is_mapped)
                .count() as u64,
            largest_alloc_bytes: pool.largest_alloc_bytes,
        }
    }
}
