mod base;

// I was going to implement something with mmap to test on the CPU but finally decided to create this
// simulated virtual storage using bytes storage because I think fits better within the purpose of this crate.
#[cfg(feature = "storage-bytes")]
mod bytes_virtual;
#[cfg(feature = "storage-bytes")]
pub use bytes_virtual::*;

pub use base::*;
