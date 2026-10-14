#[cfg(feature = "fp8")]
mod fp8_e4m3;
#[cfg(feature = "fp8")]
mod fp8_e5m2;
mod fp8_e8m0;

#[cfg(feature = "fp8")]
pub use fp8_e4m3::*;
#[cfg(feature = "fp8")]
pub use fp8_e5m2::*;
pub use fp8_e8m0::*;
