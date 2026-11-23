use super::BytesFormat;
use alloc::string::String;
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// How much memory is in use, how much is reserved on the device to serve it,
/// and how much of it is padding.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryUsage {
    /// The number of allocations currently active.
    ///
    /// This is not the number of times an actual allocation happens to create a new memory page,
    /// but really the number of active slices.
    pub allocation_count: u64,
    /// The number of bytes that are currently actually in use.
    ///
    /// This doesn't include any padding or other memory that needs to be
    /// reserved, and is the minimum amount of memory that could possible
    /// be allocated.
    pub in_use_bytes: u64,
    /// The amount of bytes used for padding memory in currently active allocations.
    pub padding_bytes: u64,
    /// The total amount of memory reserved on the device.
    ///
    /// This will be at least as much as `in_use_bytes` but in practice will
    /// be higher, as allocations reserve memory for future allocations
    /// and for padding.
    pub reserved_bytes: u64,
}

impl MemoryUsage {
    /// Both usages together, field by field.
    pub fn combine(&self, other: MemoryUsage) -> MemoryUsage {
        MemoryUsage {
            allocation_count: self.allocation_count + other.allocation_count,
            in_use_bytes: self.in_use_bytes + other.in_use_bytes,
            padding_bytes: self.padding_bytes + other.padding_bytes,
            reserved_bytes: self.reserved_bytes + other.reserved_bytes,
        }
    }
}

impl core::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // In the future it'd be nice if MemoryUsage also held some stats about say,
        // the 5 biggest allocations, to show when you an OOM.
        let usage_percentage = (self.in_use_bytes as f32 / self.reserved_bytes as f32) * 100.0;
        let padding_percentage = (self.padding_bytes as f32 / self.in_use_bytes as f32) * 100.0;
        writeln!(f, "Memory Usage Report:")?;
        writeln!(f, "  Number of allocations: {}", self.allocation_count)?;
        writeln!(
            f,
            "  Bytes in use: {}",
            BytesFormat::from_byte_count(self.in_use_bytes)
        )?;
        writeln!(
            f,
            "  Bytes used for padding: {}",
            BytesFormat::from_byte_count(self.padding_bytes)
        )?;
        writeln!(
            f,
            "  Total bytes reserved: {}",
            BytesFormat::from_byte_count(self.reserved_bytes)
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
    pub page_count: u64,
    /// The most device allocations ever held at once: for a sliced pool, the
    /// pages the workload needed at its peak.
    pub peak_page_count: u64,
    /// How many of the current pages have no device backing yet — carved
    /// while launches were discarded, and never resolved into anything that
    /// executes. They count toward `page_count`/`peak_page_count` (the plan is the
    /// *reserved* stream) while costing no device memory; `page_count -
    /// unmapped_page_count` is what this pool's pages cost the device.
    pub unmapped_page_count: u64,
    /// The largest single allocation this pool ever served, in requested
    /// (pre-padding) bytes.
    pub largest_alloc_bytes: u64,
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
    /// One entry per stream the scope covers, only ever read in order.
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
    /// as staging buffers for reads or uniform buffers for launches, only ever
    /// read in order. Empty where the runtime keeps none.
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
    /// growth left behind last, only ever read in that order.
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
    /// What the caller was doing: `model loaded`, `after the build`.
    pub label: String,
    /// The device's memory at that moment, stream by stream.
    pub report: MemoryReport,
}

impl cubecl_environment::records::Record for MemoryRecord {
    const KIND: &'static str = "memory";
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn pool(in_use_bytes: u64) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: MemoryUsage {
                allocation_count: 1,
                in_use_bytes,
                padding_bytes: 0,
                reserved_bytes: in_use_bytes,
            },
            page_count: 1,
            peak_page_count: 1,
            unmapped_page_count: 0,
            largest_alloc_bytes: in_use_bytes,
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

    fn usage(allocations: u64, bytes: u64) -> MemoryUsage {
        MemoryUsage {
            allocation_count: allocations,
            in_use_bytes: bytes,
            padding_bytes: bytes,
            reserved_bytes: bytes,
        }
    }

    #[test]
    fn combine_sums_every_field() {
        assert_eq!(usage(1, 2).combine(usage(3, 4)), usage(4, 6));
    }

    #[test]
    fn pools_usage_sums_dynamic_persistent_and_dedicated() {
        assert_eq!(stream(0, 1).pools.usage().in_use_bytes, 3);
    }

    /// A device report is the sum of its streams.
    #[test]
    fn report_usage_sums_every_stream() {
        let report = MemoryReport {
            streams: vec![stream(0, 1), stream(1, 10)],
        };

        assert_eq!(report.usage().in_use_bytes, 33);
    }

    /// The memories a runtime keeps for itself count toward the stream.
    #[test]
    fn stream_usage_counts_auxiliary_memories() {
        let mut report = stream(0, 1);
        report.auxiliary.push(AuxiliaryMemoryReport {
            name: "staging".to_string(),
            pools: stream(0, 10).pools,
        });

        assert_eq!(report.usage().in_use_bytes, 33);
    }

    #[test]
    fn display_reports_the_usage_efficiency() {
        let report = MemoryUsage {
            reserved_bytes: 4000,
            ..usage(1, 1000)
        };

        assert!(report.to_string().contains("Usage efficiency: 25.00%"));
    }
}
