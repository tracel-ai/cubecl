#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
};

/// Amount of memory in use by this allocator
/// and statistics on how much memory is reserved and
/// wasted in total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryUsage {
    /// The number of allocations currently active.
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
    /// This will be at least as much as bytes_in_use but in practice will
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

/// The managed tensor buffer handle that points to some memory segment.
/// It should not contain actual data.
pub trait MemoryHandle<Binding>: Clone + Send + Sync + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
    /// Get the binding associated to the current handle.
    fn binding(self) -> Binding;
}
