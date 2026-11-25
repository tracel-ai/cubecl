use crate::memory_management::memory_pool::{PagedLayout, padded_size};
use cubecl_ir::MemoryDeviceProperties;

/// Slack a page keeps past the largest allocation it was sized for, so that
/// allocation still fits once aligned.
const PAGE_SLACK_BYTES: u64 = 1024 * 1024;

/// The unit a page size is rounded up to.
const PAGE_GRANULE_BYTES: u64 = 1024 * 1024;

/// How the pages of an [`ArenaReserver`](super::ArenaReserver) are sized:
/// from a floor up to the largest page the device allocates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageSizing {
    /// The smallest page, so a workload of small allocations still carves
    /// pages worth carving.
    pub min_page_bytes: u64,
    /// The largest page the device allocates, alignment-rounded down: an
    /// allocation no page of that size fits is not the arena's to serve.
    pub max_page_bytes: u64,
    /// The alignment every slice starts at.
    pub alignment_bytes: u64,
}

/// How an allocation fits the pages the arena carves now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fit {
    /// A page of the current size holds it.
    Fits,
    /// It needs pages of `page_size_bytes`, which outdates the current ones.
    Grow {
        /// The size the new pages are allocated at.
        page_size_bytes: u64,
    },
    /// No page the device allocates holds it.
    TooLarge,
}

impl PageSizing {
    /// Pages from `min_page_size` up to the largest the device with
    /// `properties` allocates.
    pub(crate) fn new(min_page_size: u64, properties: &MemoryDeviceProperties) -> Self {
        let alignment = properties.alignment;
        let max_page_bytes = (properties.max_page_size / alignment * alignment).max(alignment);
        Self {
            min_page_bytes: min_page_size
                .max(alignment)
                .next_multiple_of(alignment)
                .min(max_page_bytes),
            max_page_bytes,
            alignment_bytes: alignment,
        }
    }

    /// How an allocation of `size` bytes fits pages of `page_size`.
    ///
    /// A growth sizes the new pages to the allocation, a megabyte of slack
    /// on top, rounded to the megabyte: never below the floor, never past
    /// the device's largest page.
    pub(crate) fn fit(&self, size: u64, page_size: u64) -> Fit {
        let Some(needed) = padded_size(size, self.alignment_bytes) else {
            return Fit::TooLarge;
        };
        if needed <= page_size {
            return Fit::Fits;
        }
        if needed > self.max_page_bytes {
            return Fit::TooLarge;
        }
        Fit::Grow {
            page_size_bytes: size
                .saturating_add(PAGE_SLACK_BYTES)
                .next_multiple_of(PAGE_GRANULE_BYTES)
                .next_multiple_of(self.alignment_bytes)
                .clamp(self.min_page_bytes, self.max_page_bytes),
        }
    }

    /// The layout of a pool carving pages of `page_size`, whose slices carry
    /// `index`.
    ///
    /// It takes every size it is handed, so nothing is refused for being
    /// larger than a slice of a page it fits.
    pub(crate) fn layout(&self, page_size: u64, index: u8) -> PagedLayout {
        PagedLayout {
            page_size_bytes: page_size,
            max_slice_bytes: page_size,
            alignment_bytes: self.alignment_bytes,
            pool: index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn sizing() -> PageSizing {
        PageSizing::new(2 * MIB, &MemoryDeviceProperties::new(16 * MIB, 32))
    }

    #[test]
    fn an_allocation_that_fits_the_pages_grows_nothing() {
        assert_eq!(sizing().fit(MIB, 2 * MIB), Fit::Fits);
    }

    #[test]
    fn a_growth_sizes_pages_to_the_allocation_with_slack() {
        assert_eq!(
            sizing().fit(10 * MIB, 2 * MIB),
            Fit::Grow {
                page_size_bytes: 11 * MIB
            }
        );
    }

    #[test]
    fn a_growth_stops_at_the_devices_largest_page() {
        assert_eq!(
            sizing().fit(16 * MIB, 2 * MIB),
            Fit::Grow {
                page_size_bytes: 16 * MIB
            }
        );
        assert_eq!(sizing().fit(16 * MIB + 1, 2 * MIB), Fit::TooLarge);
    }

    /// A size whose padding passes the largest `u64` is too large, rather than
    /// wrapped to one that fits.
    #[test]
    fn a_size_whose_padding_overflows_is_too_large() {
        assert_eq!(sizing().fit(u64::MAX, 2 * MIB), Fit::TooLarge);
    }

    /// A growth never sizes pages below the floor.
    #[test]
    fn a_growth_never_goes_below_the_floor() {
        let sizing = PageSizing::new(8 * MIB, &MemoryDeviceProperties::new(16 * MIB, 32));

        assert_eq!(
            sizing.fit(2 * MIB, MIB),
            Fit::Grow {
                page_size_bytes: 8 * MIB
            }
        );
    }

    /// The floor never passes the device's largest page.
    #[test]
    fn the_floor_never_passes_the_largest_page() {
        let sizing = PageSizing::new(32 * MIB, &MemoryDeviceProperties::new(16 * MIB, 32));

        assert_eq!(sizing.min_page_bytes, 16 * MIB);
    }
}
