use alloc::string::{String, ToString};
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// Amount of memory in use by this allocator
/// and statistics on how much memory is reserved and
/// wasted in total.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MemoryPoolKind {
    /// Allocations are slices carved from shared pages.
    Sliced {
        /// The size of each device page.
        page_size: u64,
        /// The largest allocation the pool accepts.
        max_slice_size: u64,
    },
    /// Slices carved from pages sized after the largest allocation served.
    Adaptive {
        /// The size new pages are allocated at.
        page_size: u64,
        /// Pages held at an older, smaller size, waiting on their last live
        /// slice before they are returned to the driver.
        outdated_pages: u64,
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryPoolReport {
    /// The pool's shape and effective configuration.
    pub kind: MemoryPoolKind,
    /// The pool's current usage.
    pub usage: MemoryUsage,
    /// Device allocations (pages) currently held.
    pub pages: u64,
    /// The most device allocations ever held at once: for a sliced pool, the
    /// pages the workload needed at its peak.
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

/// Which memory a [`MemoryReport`] covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Every stream's memory on the device.
    Device,
    /// The memory of the stream the client issues on.
    CurrentStream,
}

/// Everything the memory of a [`MemoryScope`] holds, stream by stream: the
/// single place memory is read from. A caller that wants the totals asks for
/// the [`usage`](Self::usage).
///
/// A tuning pass allocates like anything else, so its scratch counts toward
/// these marks; warming the tune caches in an earlier pass leaves the peaks of
/// a measured one to the workload alone.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    /// One entry per stream the scope covers.
    pub streams: Vec<StreamMemoryReport>,
}

/// What one stream's memory holds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamMemoryReport {
    /// The stream the memory belongs to.
    pub stream: StreamId,
    /// The pools every allocation a user makes lives in.
    pub pools: MemoryPoolsReport,
    /// The memories a runtime keeps beside those pools for its own use, such
    /// as staging buffers for reads or uniform buffers for launches. Empty
    /// where the runtime keeps none.
    pub auxiliary: Vec<AuxiliaryMemoryReport>,
}

/// A memory a runtime keeps for its own use, and what it holds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AuxiliaryMemoryReport {
    /// What the memory is for.
    pub name: String,
    /// Its pools.
    pub pools: MemoryPoolsReport,
}

/// What a memory holds, pool by pool.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryPoolsReport {
    /// One entry per dynamic pool, in allocation-routing order, the pools a
    /// growth left behind last.
    pub dynamic: Vec<MemoryPoolReport>,
    /// The persistent pool (weights, caches; explicit persistent windows).
    pub persistent: MemoryPoolReport,
    /// The dedicated allocations, each its own device allocation.
    pub dedicated: MemoryPoolReport,
}

impl MemoryReport {
    /// The usage of every pool of every stream together.
    pub fn usage(&self) -> MemoryUsage {
        self.streams
            .iter()
            .fold(MemoryUsage::default(), |usage, stream| {
                usage.combine(stream.usage())
            })
    }
}

impl StreamMemoryReport {
    /// The usage of this stream's pools together, the auxiliary memories'
    /// included.
    pub fn usage(&self) -> MemoryUsage {
        self.auxiliary
            .iter()
            .fold(self.pools.usage(), |usage, memory| {
                usage.combine(memory.pools.usage())
            })
    }
}

impl MemoryPoolsReport {
    /// The usage of every pool together.
    pub fn usage(&self) -> MemoryUsage {
        self.dynamic
            .iter()
            .chain([&self.persistent, &self.dedicated])
            .fold(MemoryUsage::default(), |usage, pool| {
                usage.combine(pool.usage.clone())
            })
    }
}

/// A [`MemoryReport`] as the environment records it: a snapshot of every
/// stream's memory on the device at a moment the caller named, written by
/// [`Client::record_memory`](crate::client::Client::record_memory).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryRecord {
    /// What the caller was doing: `model loaded`, `after the dry run`.
    pub label: alloc::string::String,
    /// The device's memory at that moment, stream by stream.
    pub report: MemoryReport,
}

impl cubecl_environment::records::Record for MemoryRecord {
    const KIND: &'static str = "memory";
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

    fn pool(bytes_in_use: u64) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: MemoryUsage {
                number_allocs: 1,
                bytes_in_use,
                bytes_padding: 0,
                bytes_reserved: bytes_in_use,
            },
            pages: 1,
            pages_peak: 1,
            pages_unmapped: 0,
            largest_alloc: bytes_in_use,
        }
    }

    fn stream(value: u64, bytes: u64) -> StreamMemoryReport {
        StreamMemoryReport {
            stream: StreamId { value },
            pools: MemoryPoolsReport {
                dynamic: vec![pool(bytes)],
                persistent: pool(bytes),
                dedicated: pool(bytes),
            },
            auxiliary: Vec::new(),
        }
    }

    /// A device report is the sum of its streams, each the sum of its pools.
    #[test]
    fn usage_sums_every_pool_of_every_stream() {
        let report = MemoryReport {
            streams: vec![stream(0, 1), stream(1, 10)],
        };
        assert_eq!(report.streams[0].usage().bytes_in_use, 3);
        assert_eq!(report.usage().bytes_in_use, 33);
        assert_eq!(report.usage().number_allocs, 6);
    }

    /// The memories a runtime keeps for itself count toward the stream.
    #[test]
    fn usage_counts_auxiliary_memories() {
        let mut report = stream(0, 1);
        report.auxiliary.push(AuxiliaryMemoryReport {
            name: "staging".to_string(),
            pools: stream(0, 10).pools,
        });
        assert_eq!(report.usage().bytes_in_use, 33);
    }
}
