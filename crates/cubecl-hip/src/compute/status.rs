//! One way to ask whether a HIP call succeeded.
//!
//! The driver reports through a status code and the caller has to turn that
//! into something the rest of the runtime understands. Three answers used to
//! coexist — a helper in one file, bare asserts in another, and a hand-written
//! error at eighteen more sites — so a failed call meant a panic, an
//! `IoError`, or a `ServerError` depending on which file it happened in.
//!
//! [`HipError`] is the one answer, and the `From` implementations are how it
//! becomes whichever error the caller's signature promises.

use cubecl_core::server::{IoError, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::compiler::CompilationError;
use std::ffi::c_uint;

/// A HIP entry point that failed, named by what was called.
///
/// The status is the driver's own code, kept as a number: the two APIs number
/// their enums differently and neither is worth a table here. Naming the entry
/// point is what makes the number searchable.
#[derive(Debug, Clone)]
pub struct HipError {
    op: &'static str,
    status: c_uint,
}

impl core::fmt::Display for HipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} failed with HIP status {}", self.op, self.status)
    }
}

impl From<HipError> for ServerError {
    fn from(error: HipError) -> Self {
        ServerError::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<HipError> for IoError {
    fn from(error: HipError) -> Self {
        IoError::Unknown {
            description: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<HipError> for CompilationError {
    fn from(error: HipError) -> Self {
        CompilationError::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

/// `Ok` when `status` says the call to `op` succeeded.
///
/// Serves both the HIP runtime and HIP RTC: the two enumerate their failures
/// differently but both report success as zero, and `op` is what tells a
/// reader which of them a number belongs to.
///
/// # Errors
///
/// [`HipError`], which `?` turns into whichever error the caller returns.
pub fn checked(op: &'static str, status: c_uint) -> Result<(), HipError> {
    match status {
        0 => Ok(()),
        status => Err(HipError { op, status }),
    }
}
