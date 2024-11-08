mod base;
mod config;

mod algorithm;

pub use algorithm::{cmma, plane_mma, Algorithm};
pub use base::{launch, launch_ref};
pub use config::{create_stage_dim, AdvancedConfig};
