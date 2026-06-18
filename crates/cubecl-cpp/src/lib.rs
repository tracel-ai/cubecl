#[macro_use]
extern crate derive_new;

pub mod error;
pub mod shared;

pub use shared::ComputeKernel;
pub use shared::register_supported_types;

/// Format CPP code.
pub mod formatter;

pub mod cuda;
pub mod hip;
#[cfg(feature = "metal")]
pub mod metal;
pub mod target;

#[cfg(feature = "metal")]
pub type MslCompiler = shared::CppCompiler<target::Metal>;
