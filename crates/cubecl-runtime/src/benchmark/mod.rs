pub use cubecl_common::benchmark::*;

#[cfg(feature = "std")]
mod multi_device;
#[cfg(feature = "std")]
pub use multi_device::*;
