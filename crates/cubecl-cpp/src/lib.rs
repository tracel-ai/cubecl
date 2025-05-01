#[macro_use]
extern crate derive_new;

pub mod shared;

pub use shared::ComputeKernel;
pub use shared::register_supported_types;
pub use shared::{Dialect, DialectWmmaCompiler};

/// Format CPP code.
pub mod formatter;

#[cfg(feature = "hip")]
pub mod hip;

// The hip dialects use the cuda dialect sometimes this is why we need it for hip feature as well
#[cfg(any(feature = "cuda", feature = "hip"))]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "metal")]
pub type MslCompiler = shared::CppCompiler<metal::MslDialect>;
