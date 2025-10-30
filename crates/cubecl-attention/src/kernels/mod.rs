/// Accelerated but using shared memory at its core
pub mod blackbox_accelerated;
/// Very slow attention implementation. Temporary
pub mod dummy;
/// Unit attention
pub mod unit;

mod algorithm;

pub use algorithm::*;
