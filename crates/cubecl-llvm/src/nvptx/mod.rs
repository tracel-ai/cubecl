//! NVPTX target.

pub mod abi;
pub mod builtins;
pub mod codegen;
pub mod libdevice;
pub mod matrix;
#[cfg(test)]
mod offline_tests;
pub mod plane;
pub mod printf;
pub mod ptx_version;
pub mod synchronization;
