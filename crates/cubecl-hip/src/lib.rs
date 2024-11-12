#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

#[cfg(target_os = "linux")]
pub mod compute;
#[cfg(target_os = "linux")]
pub mod device;
#[cfg(target_os = "linux")]
pub mod runtime;
#[cfg(target_os = "linux")]
pub use device::*;
#[cfg(target_os = "linux")]
pub use runtime::HipRuntime;
#[cfg(feature = "rocwmma")]
pub type HipDialect = cubecl_cpp::HipDialectRocWmma;
#[cfg(feature = "rocwmma")]
pub type HipCompiler = cubecl_cpp::HipCompilerRocWmma;

#[cfg(feature = "wmma_intrinsic")]
pub type HipDialect = cubecl_cpp::HipDialectInstrinsic;
#[cfg(feature = "wmma_intrinsic")]
pub type HipCompiler = cubecl_cpp::HipCompilerInstrinsic;

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::HipRuntime<HipDialect>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_cmma_old!();
}
