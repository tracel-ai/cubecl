use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Amount of memory in use by this allocator
/// and statistics on how much memory is reserved and
/// wasted in total.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// The number of allocations currently active.
    ///
    /// This is not the number of times an actual allocation happens to create a new memory page,
    /// but really the number of active slices.
    pub number_allocs: u64,
    /// The number of bytes that are currently actually in use.
    ///
    /// This doesn't include any padding or other memory that needs to be
    /// reserved, and is the minimum amount of memory that could possible
    /// be allocated.
    pub bytes_in_use: u64,
    /// The amount of bytes used for padding memory in currently active allocations.
    pub bytes_padding: u64,
    /// The total amount of memory reserved on the device.
    ///
    /// This will be at least as much as `bytes_in_use` but in practice will
    /// be higher, as allocations reserve memory for future allocations
    /// and for padding.
    pub bytes_reserved: u64,
}

impl MemoryUsage {
    /// Calculate the combined memory usage of two reports (summing them).
    pub fn combine(&self, other: MemoryUsage) -> MemoryUsage {
        MemoryUsage {
            number_allocs: self.number_allocs + other.number_allocs,
            bytes_in_use: self.bytes_in_use + other.bytes_in_use,
            bytes_padding: self.bytes_padding + other.bytes_padding,
            bytes_reserved: self.bytes_reserved + other.bytes_reserved,
        }
    }
}

#[derive(new)]
#[doc(hidden)]
pub struct BytesFormat {
    bytes: u64,
}

impl core::fmt::Display for BytesFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let unit = 1000;

        if self.bytes < unit {
            f.write_fmt(format_args!("{} B", self.bytes))
        } else {
            let size = self.bytes as f64;
            let exp = match size.log(1000.0).floor() as usize {
                0 => 1,
                e => e,
            };
            let unit_prefix = "KMGTPEZY".as_bytes();
            f.write_fmt(format_args!(
                "{:.2} {}B",
                (size / unit.pow(exp as u32) as f64),
                unit_prefix[exp - 1] as char,
            ))
        }
    }
}

fn bytes_format(bytes: u64) -> String {
    BytesFormat::new(bytes).to_string()
}

impl core::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // In the future it'd be nice if MemoryUsage also held some stats about say,
        // the 5 biggest allocations, to show when you an OOM.
        let usage_percentage = (self.bytes_in_use as f32 / self.bytes_reserved as f32) * 100.0;
        let padding_percentage = (self.bytes_padding as f32 / self.bytes_in_use as f32) * 100.0;
        writeln!(f, "Memory Usage Report:")?;
        writeln!(f, "  Number of allocations: {}", self.number_allocs)?;
        writeln!(f, "  Bytes in use: {}", bytes_format(self.bytes_in_use))?;
        writeln!(
            f,
            "  Bytes used for padding: {}",
            bytes_format(self.bytes_padding)
        )?;
        writeln!(
            f,
            "  Total bytes reserved: {}",
            bytes_format(self.bytes_reserved)
        )?;
        writeln!(f, "  Usage efficiency: {usage_percentage:.2}%")?;
        writeln!(f, "  Padding overhead: {padding_percentage:.2}%")
    }
}

/// The pool shape a [`MemoryPoolReport`] describes, carrying the pool's
/// effective configuration (after alignment rounding and page-size shrinking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPoolKind {
    /// Allocations are slices carved from shared pages.
    Sliced {
        /// The size of each device page.
        page_size: u64,
        /// The largest allocation the pool accepts.
        max_slice_size: u64,
        /// The pool's byte cap (`None` grows unbounded).
        max_pool_size: Option<u64>,
    },
    /// Every allocation is its own device page.
    Exclusive {
        /// The largest allocation the pool accepts.
        max_alloc_size: u64,
    },
    /// One device allocation per reservation, sized to the request, reused by
    /// exact size and returned to the driver only under memory pressure.
    /// Wastes only alignment padding, and pays a driver allocation per
    /// distinct size rather than per page.
    Direct,
    /// Exact-fit slices that are reused only by identical size.
    Persistent,
}

/// A structured snapshot of one memory pool: its shape, its current usage, and
/// the high-water marks a memory plan is derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPoolReport {
    /// The pool's shape and effective configuration.
    pub kind: MemoryPoolKind,
    /// The pool's current usage.
    pub usage: MemoryUsage,
    /// Device allocations (pages) currently held.
    pub pages: u64,
    /// The most device allocations ever held at once.
    ///
    /// For a sliced pool this is the number a capped layout needs:
    /// pages are carved by a deterministic first-fit policy, so replaying the
    /// same allocation stream against `pages_peak * page_size` fits by
    /// construction.
    pub pages_peak: u64,
    /// How many of the current pages have no device backing yet — carved
    /// under a dry run and never resolved into anything that executes. They
    /// count toward `pages`/`pages_peak` (the plan is the *reserved* stream)
    /// while costing no device memory; `pages - pages_unmapped` is the dry
    /// run's actual footprint in this pool.
    pub pages_unmapped: u64,
    /// The largest single allocation this pool ever served, in requested
    /// (pre-padding) bytes.
    pub largest_alloc: u64,
}

/// A per-pool report of one `MemoryManagement` (in `cubecl-server`)
/// instance — the read side of a measured memory plan.
///
/// The intended cycle: install a growable layout, run the workload once under
/// a [`DryRun`](crate::dry_run::DryRun) (same allocation stream, no compute),
/// read this report, and re-install the same layout capped at the observed
/// `pages_peak`. Padding then comes only from alignment and the first-fit
/// remainders the dry run already measured.
///
/// A tuning pass inside the measured run allocates too, and its scratch counts
/// toward these marks like anything else. Warming the tune caches in an
/// earlier pass and rebuilding the pools
/// (`MemoryManagement::install_pools`, which resets the
/// marks)
/// before the measured one leaves the peaks to the workload alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReport {
    /// One entry per dynamic pool, in allocation-routing order — the same
    /// order the layout was configured with.
    pub dynamic: Vec<MemoryPoolReport>,
    /// The persistent pool (weights, caches; explicit persistent windows).
    pub persistent: MemoryPoolReport,
}

impl MemoryPoolKind {
    /// The largest size each of `allocs` equally sized allocations may take
    /// while all of them are held, or `None` where the pool accepts any size.
    ///
    /// A capped pool holds whole pages and a slice never spans two, so what
    /// bounds a set of equal allocations is how many of them a page fits:
    /// `ceil(allocs / pages)` share one, and each may take that fraction of it.
    /// Alignment padding is not modelled, so a request at the ceiling can still
    /// pad past it, as can an allocation of another size held alongside.
    fn max_servable(&self, allocs: u64) -> Option<u64> {
        match *self {
            // Every reservation is sized to the request.
            Self::Direct => None,
            // An uncapped sliced pool grows a page per allocation and takes
            // near-page-size strays, so its page is the ceiling. A capped one
            // routes on slice size, within the pages it may hold.
            Self::Sliced {
                page_size,
                max_slice_size,
                max_pool_size,
            } => Some(match max_pool_size {
                None => page_size,
                Some(cap) => {
                    let pages = (cap / page_size.max(1)).max(1);

                    max_slice_size.min(page_size / allocs.div_ceil(pages))
                }
            }),
            Self::Exclusive { max_alloc_size } => Some(max_alloc_size),
            // Reserved for an explicit persistent window, never routed here.
            Self::Persistent => Some(0),
        }
    }
}

impl MemoryReport {
    /// The largest size each of `allocs` equally sized allocations may take
    /// while all of them are held by the installed dynamic layout, or `None`
    /// where a pool accepts any size.
    ///
    /// This is what an allocation is routed against, unlike
    /// [`MemoryDeviceProperties::max_page_size`], which sizes the default
    /// layouts and says nothing about an installed one. A caller free to pick
    /// its own size asks this, so a workload-sized layout narrows it instead of
    /// refusing it.
    ///
    /// A capped pool bounds what it holds at once, not what it accepts once,
    /// so a caller keeping several buffers live asks for `allocs` of them. Any
    /// allocation of a *different* size held alongside is the caller's to make
    /// room for.
    pub fn max_servable(&self, allocs: u64) -> Option<u64> {
        let allocs = allocs.max(1);
        let mut max = 0;

        // An allocation is offered to each pool in turn, so the most permissive
        // one sets the ceiling.
        for pool in &self.dynamic {
            max = max.max(pool.kind.max_servable(allocs)?);
        }

        Some(max)
    }
}

/// The managed tensor buffer handle that points to some memory segment.
/// It should not contain actual data.
pub trait MemoryHandle<Binding>: Clone + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
    /// Get the binding associated to the current handle.
    fn binding(self) -> Binding;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const MB: u64 = 1024 * 1024;

    fn pool(kind: MemoryPoolKind) -> MemoryPoolReport {
        MemoryPoolReport {
            kind,
            usage: MemoryUsage::default(),
            pages: 0,
            pages_peak: 0,
            pages_unmapped: 0,
            largest_alloc: 0,
        }
    }

    fn report(dynamic: Vec<MemoryPoolReport>) -> MemoryReport {
        MemoryReport {
            dynamic,
            persistent: pool(MemoryPoolKind::Persistent),
        }
    }

    /// The ceiling is the most permissive pool.
    #[test]
    fn the_widest_pool_sets_the_ceiling() {
        let layout = report(vec![
            pool(MemoryPoolKind::Sliced {
                page_size: 8 * MB,
                max_slice_size: 64 * 1024,
                max_pool_size: None,
            }),
            pool(MemoryPoolKind::Exclusive {
                max_alloc_size: 256 * MB,
            }),
        ]);

        assert_eq!(layout.max_servable(1), Some(256 * MB));
    }

    /// An uncapped sliced pool takes an allocation the size of its page, where
    /// a capped one takes nothing above its slice.
    #[test]
    fn a_cap_holds_a_sliced_pool_to_its_slice() {
        let uncapped = report(vec![pool(MemoryPoolKind::Sliced {
            page_size: 513 * MB,
            max_slice_size: 64 * MB,
            max_pool_size: None,
        })]);
        assert_eq!(uncapped.max_servable(1), Some(513 * MB));

        let capped = report(vec![pool(MemoryPoolKind::Sliced {
            page_size: 513 * MB,
            max_slice_size: 64 * MB,
            max_pool_size: Some(2 * 513 * MB),
        })]);
        assert_eq!(capped.max_servable(1), Some(64 * MB));
    }

    /// Allocations a capped pool cannot spread over pages of its own share a
    /// page, so each may take a fraction of one.
    #[test]
    fn allocations_beyond_the_pages_share_one() {
        let layout = |pages: u64| {
            report(vec![pool(MemoryPoolKind::Sliced {
                page_size: 512 * MB,
                max_slice_size: 512 * MB,
                max_pool_size: Some(pages * 512 * MB),
            })])
        };

        // A page each, so the slice limit stands.
        assert_eq!(layout(2).max_servable(2), Some(512 * MB));
        // Both in one page, so each takes half of it.
        assert_eq!(layout(1).max_servable(2), Some(256 * MB));
        assert_eq!(layout(1).max_servable(3), Some(512 * MB / 3));
        // Three across two pages: two of them share.
        assert_eq!(layout(2).max_servable(3), Some(256 * MB));
        // Holding nothing is holding one.
        assert_eq!(layout(1).max_servable(0), layout(1).max_servable(1));
    }

    /// An uncapped pool grows a page per allocation, so holding several
    /// narrows nothing.
    #[test]
    fn an_uncapped_pool_serves_any_number() {
        let layout = report(vec![
            pool(MemoryPoolKind::Sliced {
                page_size: 512 * MB,
                max_slice_size: 64 * MB,
                max_pool_size: None,
            }),
            pool(MemoryPoolKind::Exclusive {
                max_alloc_size: 256 * MB,
            }),
        ]);

        assert_eq!(layout.max_servable(8), Some(512 * MB));
    }

    /// A direct pool sizes every reservation to the request, so the layout
    /// names no ceiling at all.
    #[test]
    fn a_direct_pool_has_no_ceiling() {
        let layout = report(vec![
            pool(MemoryPoolKind::Exclusive {
                max_alloc_size: 64 * MB,
            }),
            pool(MemoryPoolKind::Direct),
        ]);

        assert_eq!(layout.max_servable(1), None);
    }

    /// The persistent pool serves explicit windows, not the routing an
    /// ordinary allocation takes, so it never raises the ceiling.
    #[test]
    fn the_persistent_pool_is_not_a_dynamic_ceiling() {
        let layout = report(vec![
            pool(MemoryPoolKind::Exclusive {
                max_alloc_size: 64 * MB,
            }),
            pool(MemoryPoolKind::Persistent),
        ]);

        assert_eq!(layout.max_servable(1), Some(64 * MB));
        assert_eq!(report(vec![]).max_servable(1), Some(0));
    }
}
