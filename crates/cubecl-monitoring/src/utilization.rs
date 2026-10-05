use alloc::string::String;
use core::fmt;

use cubecl_ir::PciVendor;

/// How busy a device is, as its counter read it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DeviceUtilization {
    /// The percent of the counter's last period the device spent running work, from 0 to 100.
    pub busy_percent: f32,
}

impl DeviceUtilization {
    /// Holds a reading outside 0 to 100 to the nearer bound.
    pub fn new(busy_percent: f32) -> Self {
        Self {
            busy_percent: busy_percent.clamp(0.0, 100.0),
        }
    }
}

/// Why a device's utilization could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UtilizationUnavailable {
    /// This build does not link the device's runtime, which is what says which card the device
    /// is.
    RuntimeNotLinked,
    /// The device's runtime reads no counter for its devices.
    RuntimeReadsNoCounter,
    /// The runtime reports no card behind the device: a software adapter.
    NoCard,
    /// This platform keeps no counter read here for a card of this vendor, or of no vendor the
    /// runtime named.
    NoCounterForCard(Option<PciVendor>),
    /// The card's counter is found by an address the runtime did not report: the PCI address on
    /// Linux, the LUID on Windows, and on macOS the registry entry id, where the machine has more
    /// than one GPU.
    CardAddressNotReported,
    /// The driver's library would not load: no driver installed, or none on the loader's path.
    /// Holds the loader's message.
    DriverLibraryNotFound(String),
    /// The counter measures the time between two readings, and this was the device's first, or
    /// the first since the counter restarted.
    NoPreviousReading,
    /// The counter was reached and gave no reading. Holds the driver's, the platform's or the
    /// filesystem's message.
    QueryFailed(String),
}

impl fmt::Display for UtilizationUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeNotLinked => f.write_str("this build does not link the device's runtime"),
            Self::RuntimeReadsNoCounter => {
                f.write_str("the device's runtime reads no utilization counter")
            }
            Self::NoCard => f.write_str("the device is no card: a software adapter"),
            Self::NoCounterForCard(Some(vendor)) => {
                write!(f, "no counter on this platform for a {vendor} card")
            }
            Self::NoCounterForCard(None) => {
                f.write_str("no counter on this platform for a card of no known vendor")
            }
            Self::CardAddressNotReported => {
                f.write_str("the runtime did not report the address the card's counter is found by")
            }
            Self::DriverLibraryNotFound(message) => {
                write!(f, "the driver's library would not load: {message}")
            }
            Self::NoPreviousReading => {
                f.write_str("the counter measures between two readings, and this was the first")
            }
            Self::QueryFailed(message) => write!(f, "the counter gave no reading: {message}"),
        }
    }
}

impl core::error::Error for UtilizationUnavailable {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reading_from_0_to_100_is_kept_as_it_is() {
        for busy_percent in [0.0, 37.5, 100.0] {
            assert_eq!(
                DeviceUtilization::new(busy_percent).busy_percent,
                busy_percent
            );
        }
    }

    #[test]
    fn a_reading_above_100_is_held_to_100() {
        for busy_percent in [100.5, f32::MAX] {
            assert_eq!(DeviceUtilization::new(busy_percent).busy_percent, 100.0);
        }
    }
}
