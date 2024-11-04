mod base;
mod config;
mod dispatch;

pub use base::{launch, launch_ref};

#[cfg(feature = "export_tests")]
pub use {
    config::{make_cmma_config, AdvancedConfig},
    dispatch::MatmulLaunchDispatch,
};
