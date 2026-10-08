//! NVPTX target.

pub mod abi;
pub(crate) mod address;
pub mod barrier;
pub mod builtins;
pub mod codegen;
pub(crate) mod inline_asm;
pub mod libdevice;
pub mod matrix;
#[cfg(test)]
mod offline_tests;
pub mod plane;
pub mod printf;
pub mod ptx_version;
pub(crate) mod registers;
pub mod synchronization;
pub mod tma;
pub mod wgmma;
