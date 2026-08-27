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

/// A driver entry point that failed, named by what was called.
///
/// The status is kept as a number rather than decoded: each API numbers its
/// own enum and neither table belongs here. Naming the entry point is what
/// makes the number searchable in the vendor's headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverError {
    op: &'static str,
    status: u32,
}

impl DriverError {
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
        ServerError::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<DriverError> for IoError {
    fn from(error: DriverError) -> Self {
        IoError::Unknown {
            description: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<DriverError> for CompilationError {
    fn from(error: DriverError) -> Self {
        CompilationError::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<DriverError> for LaunchError {
    fn from(error: DriverError) -> Self {
        LaunchError::Unknown {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
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
        status => Err(DriverError { op, status }),
    }
}
