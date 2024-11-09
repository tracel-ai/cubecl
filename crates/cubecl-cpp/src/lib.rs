#[macro_use]
extern crate derive_new;

mod shared;

pub use shared::register_supported_types;
pub use shared::Dialect;

/// Format CPP code.
pub mod formatter;

#[cfg(feature = "hip")]
mod hip;
#[cfg(feature = "hip")]
pub type HipCompilerInstrinsic = shared::CppCompiler<hip::HipDialect<hip::WmmaIntrinsicCompiler>>;
#[cfg(feature = "hip")]
pub type HipCompilerRocWmma = shared::CppCompiler<hip::HipDialect<hip::RocWmmaCompiler>>;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub type CudaCompiler = shared::CppCompiler<cuda::CudaDialect<cuda::CudaWmmaCompiler>>;
