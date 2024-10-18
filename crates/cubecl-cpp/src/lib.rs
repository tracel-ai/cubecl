#[macro_use]
extern crate derive_new;

mod shared;

/// Format CPP code.
pub mod formatter;

#[cfg(feature = "hip")]
mod hip;
#[cfg(feature = "hip")]
pub type HipCompiler = shared::CppCompiler<hip::Hip>;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub type CudaCompiler = shared::CppCompiler<cuda::Cuda>;
