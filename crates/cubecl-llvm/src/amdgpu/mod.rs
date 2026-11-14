//! AMDGPU target.

pub mod abi;
pub mod builtins;
pub mod codegen;
pub mod device_libs;
mod fragment;
pub mod intrinsic;
pub mod lld;
pub mod matrix;
pub mod mfma;
pub mod ocml;
#[cfg(test)]
mod offline_tests;
pub mod plane;
pub mod printf;
pub mod synchronization;
