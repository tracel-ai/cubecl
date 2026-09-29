//! The one payload every error type carries when the device is lost/poisoned.
//!
//! A poisoning fault can surface from any call, and the calls do not return the
//! same error: a read returns an [`IoError`](crate::server::IoError), a
//! compilation a [`CompilationError`](crate::compiler::CompilationError), a
//! launch a [`LaunchError`](crate::server::LaunchError). So each of them has a
//! `DevicePoisoned` variant — but the payload, and the question a caller asks of
//! it, are the same everywhere, and live here.

use alloc::string::String;
use cubecl_environment::backtrace::BackTrace;
use thiserror::Error;

/// A fault the driver makes sticky: an illegal address, a trap, an ECC error, a
/// device lost outright. Nothing on the device can be trusted afterwards, and
/// every later call on it fails the same way until the process exits.
///
/// Carried by every error type rather than spelled out by each of them, so the
/// answer is the same whichever call reported it. `#[from]` on each variant is
/// what makes that ergonomic: a backend builds one of these and `?` lands it in
/// whichever error its signature promises.
#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[error("{reason}\nBacktrace:\n{backtrace}")]
pub struct DevicePoison {
    /// The driver call that reported it.
    pub reason: String,
    /// The backtrace for this error.
    #[cfg_attr(serializable, serde(skip))]
    pub backtrace: BackTrace,
}

impl DevicePoison {
    /// The fault, with the backtrace captured here.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl core::fmt::Debug for DevicePoison {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
