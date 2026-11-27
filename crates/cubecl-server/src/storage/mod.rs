//! Storage implementations. The [`StorageAllocator`] contract is
//! `cubecl-runtime`'s and re-exported here.

pub use cubecl_runtime::storage::*;

mod host_importer;
mod host_imports;
mod host_memory_api;
#[cfg(test)]
mod host_memory_fake;
mod host_pinner;
mod host_pins;
mod pinned;
pub use host_imports::*;
pub use host_memory_api::*;
pub use host_pins::*;
pub use pinned::*;

#[cfg(feature = "storage-bytes")]
mod bytes_cpu;
#[cfg(feature = "storage-bytes")]
pub use bytes_cpu::*;
