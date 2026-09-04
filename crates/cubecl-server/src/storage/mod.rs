//! Storage implementations. The [`ComputeStorage`] contract is
//! `cubecl-runtime`'s and re-exported here.

pub use cubecl_runtime::storage::*;

mod pinned;
pub use pinned::*;

#[cfg(feature = "storage-bytes")]
mod bytes_cpu;
#[cfg(feature = "storage-bytes")]
pub use bytes_cpu::*;
