mod base;
mod config;

mod algorithm;

pub use algorithm::{cmma, plane_mma, Algorithm};
pub use base::{launch, launch_ref};

#[cfg(feature = "export_tests")]
pub use config::AdvancedConfig;
