mod base;
mod virtual_storage;

pub use base::*;

#[cfg(feature = "storage-bytes")]
mod bytes_cpu;
#[cfg(feature = "storage-bytes")]
pub use bytes_cpu::*;

pub use virtual_storage::*;
