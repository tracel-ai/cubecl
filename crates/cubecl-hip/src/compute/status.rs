//! HIP's status codes, as the runtime's [`DriverError`].
//!
//! The shared [`cubecl_server::driver::checked`] cannot tell which codes poison
//! the device, because each API numbers its own enum. This is the same check
//! with HIP's answer: a fault the runtime makes sticky poisons the context,
//! every later call on it fails the same way, and the error has to say so, or
//! a caller would keep retrying on a dead device.

use cubecl_hip_sys::{
    hipError_t, hipError_t_hipErrorAssert, hipError_t_hipErrorContextIsDestroyed,
    hipError_t_hipErrorECCNotCorrectable, hipError_t_hipErrorIllegalAddress,
    hipError_t_hipErrorLaunchFailure, hipError_t_hipErrorLaunchTimeOut,
};
use cubecl_server::driver::DriverError;

/// `Ok` when `status` says the call to `op` succeeded, and a [`DriverError`]
/// flagged as poisoning the device when the status is one HIP makes sticky.
pub(crate) fn checked(op: &'static str, status: hipError_t) -> Result<(), DriverError> {
    match status {
        0 => Ok(()),
        status if poisons_device(status) => Err(DriverError::poisoned(op, status)),
        status => Err(DriverError::new(op, status)),
    }
}

/// Whether `status` is one of the faults after which the context cannot be
/// used again: the device-side counterparts of CUDA's sticky errors, which
/// HIP numbers the same way.
#[allow(
    non_upper_case_globals,
    reason = "the constants are named by the bindings"
)]
pub(crate) fn poisons_device(status: hipError_t) -> bool {
    matches!(
        status,
        hipError_t_hipErrorECCNotCorrectable
            | hipError_t_hipErrorIllegalAddress
            | hipError_t_hipErrorLaunchTimeOut
            | hipError_t_hipErrorContextIsDestroyed
            | hipError_t_hipErrorAssert
            | hipError_t_hipErrorLaunchFailure
    )
}
