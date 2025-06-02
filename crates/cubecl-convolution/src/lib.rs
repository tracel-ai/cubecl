#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil)]

mod config;

pub mod algorithm;
pub mod args;
pub mod base;
pub mod error;
pub mod homogeneous;
pub mod launch;
pub mod loader;
pub mod reader;
pub mod selection;
#[cfg(feature = "export_tests")]
pub mod tests;

pub use config::*;
pub use error::*;
pub use launch::*;
