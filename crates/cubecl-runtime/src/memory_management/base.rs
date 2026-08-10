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
pub(crate) struct BytesFormat {
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
    /// For a sliced pool this is the number a capped re-configuration needs:
    /// pages are carved by a deterministic first-fit policy, so replaying the
    /// same allocation stream against `pages_peak * page_size` fits by
    /// construction.
    pub pages_peak: u64,
    /// The largest single allocation this pool ever served, in requested
    /// (pre-padding) bytes.
    pub largest_alloc: u64,
}

/// A per-pool report of one [`MemoryManagement`](super::MemoryManagement)
/// instance — the read side of a measured memory plan.
///
/// The intended cycle: install a growable layout, run the workload once under
/// a [`DryRun`](crate::dry_run::DryRun) (same allocation stream, no compute),
/// read this report, and re-install the same layout capped at the observed
/// `pages_peak`. Padding then comes only from alignment and the first-fit
/// remainders the dry run already measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReport {
    /// One entry per dynamic pool, in allocation-routing order — the same
    /// order the layout was configured with.
    pub dynamic: Vec<MemoryPoolReport>,
    /// The persistent pool (weights, caches; explicit persistent windows).
    pub persistent: MemoryPoolReport,
    /// The dry-run pool, present when a dry run routed a measurement's
    /// allocations away from the dynamic pools (autotune scratch — see
    /// `DryRunPool`). Never part of a derived plan: measurements are already
    /// cached when the plan is replayed.
    pub dry_run: Option<MemoryPoolReport>,
}

/// The managed tensor buffer handle that points to some memory segment.
/// It should not contain actual data.
pub trait MemoryHandle<Binding>: Clone + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
    /// Get the binding associated to the current handle.
    fn binding(self) -> Binding;
}
