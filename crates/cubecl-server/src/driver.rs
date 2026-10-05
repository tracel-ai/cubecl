//! Turning a driver's status code into an error the runtime understands.
//!
//! The C-family device APIs — the CUDA and HIP runtimes, and the JIT compilers
//! beside them — all answer the same way: an integer, zero for success, an
//! enum of their own otherwise. Every backend over one of them has to turn
//! that into whichever error its caller expects, and doing it by hand at each
//! call site produces one wording per site and, sooner or later, a panic where
//! the neighbours report.
//!
//! [`checked`] is the one answer, and the `From` implementations are how a
//! `?` turns it into whichever error the caller's signature already promises.

use crate::compiler::CompilationError;
use crate::server::{IoError, LaunchError, ServerError};
use alloc::string::ToString;
use cubecl_environment::backtrace::BackTrace;

/// The payload every error type carries once the device is poisoned, re-exported.
pub use cubecl_runtime::poison::DevicePoison;

/// A driver entry point that failed, named by what was called.
///
/// The status is kept as a number rather than decoded: each API numbers its
/// own enum and neither table belongs here. Naming the entry point is what
/// makes the number searchable in the vendor's headers.
///
/// A backend also says whether the status poisoned the device: some faults (e.g. an
/// illegal address) leave the context unusable, and every later call on it fails with
/// the same code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverError {
    op: &'static str,
    status: u32,
    poisoned: bool,
}

impl DriverError {
    /// A failure a binding already turned into an error type of its own, named
    /// by the entry point that produced it.
    ///
    /// [`checked`] is the answer wherever the status is still a number. A
    /// binding that returns a typed result — cudarc's `CUresult` wrapper — has
    /// already consumed the number, and this is how the entry point's name is
    /// put back on it.
    pub fn new(op: &'static str, status: u32) -> Self {
        Self {
            op,
            status,
            poisoned: false,
        }
    }

    /// A failure rendering the device unusable.
    pub fn poisoned(op: &'static str, status: u32) -> Self {
        Self {
            op,
            status,
            poisoned: true,
        }
    }

    /// Whether this failure poisoned the device.
    pub fn is_device_poisoned(&self) -> bool {
        self.poisoned
    }

    /// The entry point that failed.
    pub fn op(&self) -> &'static str {
        self.op
    }

    /// The driver's own status code.
    pub fn status(&self) -> u32 {
        self.status
    }
}

impl core::fmt::Display for DriverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} failed with status {}", self.op, self.status)
    }
}

impl core::error::Error for DriverError {}

impl From<DriverError> for ServerError {
    fn from(error: DriverError) -> Self {
        let reason = error.to_string();
        match error.poisoned {
            true => DevicePoison::new(reason).into(),
            false => ServerError::Generic {
                reason,
                backtrace: BackTrace::capture(),
            },
        }
    }
}

impl From<DriverError> for IoError {
    fn from(error: DriverError) -> Self {
        let reason = error.to_string();
        match error.poisoned {
            true => DevicePoison::new(reason).into(),
            false => IoError::Unknown {
                description: reason,
                backtrace: BackTrace::capture(),
            },
        }
    }
}

impl From<DriverError> for CompilationError {
    fn from(error: DriverError) -> Self {
        let reason = error.to_string();
        match error.poisoned {
            true => DevicePoison::new(reason).into(),
            false => CompilationError::Generic {
                reason,
                backtrace: BackTrace::capture(),
            },
        }
    }
}

impl From<DriverError> for LaunchError {
    fn from(error: DriverError) -> Self {
        let reason = error.to_string();
        match error.poisoned {
            true => DevicePoison::new(reason).into(),
            false => LaunchError::Unknown {
                reason,
                backtrace: BackTrace::capture(),
            },
        }
    }
}

/// `Ok` when `status` says the call to `op` succeeded.
///
/// Success is zero, which the runtime and the JIT compiler of both the CUDA
/// and HIP families agree on even though their failures are numbered
/// differently. `op` is what tells a reader which numbering a code belongs to.
///
/// # Errors
///
/// [`DriverError`], which `?` turns into whichever error the caller returns.
pub fn checked(op: &'static str, status: u32) -> Result<(), DriverError> {
    match status {
        0 => Ok(()),
        status => Err(DriverError::new(op, status)),
    }
}
