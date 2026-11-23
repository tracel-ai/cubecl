use core::fmt;

/// The step between two decimal unit prefixes.
const PREFIX_STEP: u64 = 1000;

/// The decimal unit prefixes from kilo up, in order.
const PREFIXES: &[u8] = b"KMGTPEZY";

/// A byte count, displayed with the largest decimal unit that keeps it at or
/// above one (`999 B`, `1.50 KB`, `2.00 GB`).
#[doc(hidden)]
pub struct BytesFormat {
    bytes: u64,
}

impl BytesFormat {
    /// The display of `bytes` bytes.
    pub fn from_byte_count(bytes: u64) -> Self {
        Self { bytes }
    }
}

impl fmt::Display for BytesFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_bytes(f, self.bytes)
    }
}

fn write_bytes(f: &mut fmt::Formatter<'_>, bytes: u64) -> fmt::Result {
    if bytes < PREFIX_STEP {
        return write!(f, "{bytes} B");
    }
    let size = bytes as f64;
    // At least one: `log` may land just under a power of the step.
    let exp = (size.log(PREFIX_STEP as f64).floor() as usize).max(1);
    let scaled = size / PREFIX_STEP.pow(exp as u32) as f64;
    write!(f, "{scaled:.2} {}B", PREFIXES[exp - 1] as char)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn under_a_kilobyte_is_written_in_bytes() {
        assert_eq!(BytesFormat::from_byte_count(999).to_string(), "999 B");
    }

    #[test]
    fn a_kilobyte_is_written_with_its_prefix() {
        assert_eq!(BytesFormat::from_byte_count(1000).to_string(), "1.00 KB");
    }

    #[test]
    fn a_large_count_takes_the_largest_prefix_under_it() {
        assert_eq!(
            BytesFormat::from_byte_count(2_500_000_000).to_string(),
            "2.50 GB"
        );
    }
}
