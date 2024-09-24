mod base;
mod predefined;
mod strategy;

pub(crate) use base::{CmmaConfig, ComptimeCmmaInfo};
#[cfg(feature = "export_tests")]
pub(crate) use predefined::PredefinedCmmaConfig;
