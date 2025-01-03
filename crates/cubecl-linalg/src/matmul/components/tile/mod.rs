pub mod accelerated;
#[cfg(any(test, feature = "export_tests"))]
/// Use plane operations to simulate tensor cores.
///
/// Only use in testing, since it is very slow.
pub mod plane;

mod base;

pub use base::*;
