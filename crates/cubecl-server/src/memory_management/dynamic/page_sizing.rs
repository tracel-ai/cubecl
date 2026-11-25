use crate::memory_management::memory_pool::{SlicedLayout, SlicedPool, calculate_padding};
use cubecl_ir::MemoryDeviceProperties;

/// Slack a page keeps past the largest allocation it was sized for, so that
/// allocation still fits once aligned.
const PAGE_SLACK: u64 = 1024 * 1024;

/// The unit a page size is rounded up to.
const PAGE_GRANULE: u64 = 1024 * 1024;

/// How the pages of a [`PoolArena`](super::PoolArena) are sized.
#[derive(Debug, Clone, Copy)]
pub struct PageSizing {
    /// The smallest page, so a workload of small allocations still carves
    /// pages worth carving.
    pub(super) min_page_size: u64,
    /// The largest page the device allocates, alignment-rounded down: an
    /// allocation no page of that size fits is not the arena's to serve.
    max_page_size: u64,
    alignment: u64,
}

/// How an allocation fits the pages the arena carves now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// A page of the current size holds it.
    Fits,
    /// It needs pages of `page_size`, which outdates the current ones.
    Grow {
        /// The size the new pages are allocated at.
        page_size: u64,
    },
    /// No page the device allocates holds it.
    TooLarge,
}

impl PageSizing {
    /// Pages from `min_page_size` up to the largest the device with
    /// `properties` allocates.
    pub fn new(min_page_size: u64, properties: &MemoryDeviceProperties) -> Self {
        let alignment = properties.alignment;
        let max_page_size = (properties.max_page_size / alignment * alignment).max(alignment);
        Self {
            min_page_size: min_page_size
                .max(alignment)
                .next_multiple_of(alignment)
                .min(max_page_size),
            max_page_size,
            alignment,
        }
    }

    /// How an allocation of `size` bytes fits pages of `page_size`.
    ///
    /// A growth sizes the new pages to the allocation, a megabyte of slack
    /// on top, rounded to the megabyte: never below the floor, never past
    /// the device's largest page.
    pub fn fit(&self, size: u64, page_size: u64) -> Fit {
        let needed = size + calculate_padding(size, self.alignment);
        if needed <= page_size {
            return Fit::Fits;
        }
        if needed > self.max_page_size {
            return Fit::TooLarge;
        }
        Fit::Grow {
            page_size: size
                .saturating_add(PAGE_SLACK)
                .next_multiple_of(PAGE_GRANULE)
                .next_multiple_of(self.alignment)
                .clamp(self.min_page_size, self.max_page_size),
        }
    }

    /// A pool carving pages of `page_size`, whose slices carry `index`.
    pub(super) fn pool(&self, page_size: u64, index: u8) -> SlicedPool {
        // It takes every size it is handed, so nothing is refused for being
        // larger than a slice of a page it fits.
        SlicedPool::new(SlicedLayout {
            page_size,
            max_slice_size: page_size,
            alignment: self.alignment,
            pool: index,
        })
    }
}
