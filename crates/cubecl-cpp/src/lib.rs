#[macro_use]
extern crate derive_new;

pub mod shared;

pub use shared::register_supported_types;
pub use shared::{Dialect, WmmaCompiler};

/// Format CPP code.
pub mod formatter;

#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "hip")]
pub type HipDialectIntrinsic = hip::HipDialect<hip::wmma::WmmaIntrinsicCompiler>;
#[cfg(feature = "hip")]
pub type HipCompilerInstrinsic = shared::CppCompiler<HipDialectIntrinsic>;
#[cfg(feature = "hip")]
pub type HipDialectRocWmma = hip::HipDialect<hip::wmma::RocWmmaCompiler>;
#[cfg(feature = "hip")]
pub type HipCompilerRocWmma = shared::CppCompiler<HipDialectRocWmma>;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub type CudaCompiler = shared::CppCompiler<cuda::CudaDialect<cuda::wmma::CudaWmmaCompiler>>;
