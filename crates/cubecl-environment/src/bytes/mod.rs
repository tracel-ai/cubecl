mod aligned;
pub(crate) mod default_controller;
#[cfg(feature = "std")]
pub(crate) mod file;
#[cfg(feature = "std")]
mod file_reads;
#[cfg(feature = "std")]
mod held_file;
#[cfg(feature = "std")]
mod opened_file;
#[cfg(all(test, feature = "std", not(miri)))]
mod opened_file_tests;
#[cfg(feature = "shared-bytes")]
mod shared;
mod shared_arc;

mod access;
mod base;

pub use access::*;
pub use base::*;
#[cfg(feature = "std")]
pub use file_reads::FileReadCounter;
#[cfg(feature = "std")]
pub use opened_file::OpenedFile;
#[cfg(feature = "shared-bytes")]
pub use shared::SharedBytesAllocationController;
pub use shared_arc::SharedAllocationController;
