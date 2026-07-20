pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

/// Ship pre-warmed autotune and compilation caches with an application.
///
/// Run the application once so its caches are warm, then [`export`] them to a
/// file and ship it. On the target machine [`import`] fills the local
/// environment, after which the file can be deleted: nothing consults it at
/// runtime.
///
/// The exporting binary must be built against the same cubecl version as the
/// consuming one. The version is part of every cache namespace, so a bundle
/// built elsewhere installs cleanly and is then ignored, with a warning as the
/// only signal. Calling this from your own crate is what keeps the two in
/// step.
///
/// ```no_run
/// use cubecl::bundle::{export, ExportOptions};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // ... run the work you want tuned and compiled, then:
/// let roots = vec![cubecl::environment::root()];
/// let manifest = export(&roots, "h100.bundle", &ExportOptions {
///     name: "H100 Linux".to_string(),
///     ..Default::default()
/// })?;
/// println!("exported {}", manifest.name);
/// # Ok(())
/// # }
/// ```
///
/// [`export`]: cubecl_environment::bundle::export
/// [`import`]: cubecl_environment::bundle::import
pub use cubecl_environment::bundle;

/// Which named environment caches are warmed into, and where it lives.
pub use cubecl_environment::environment;

#[cfg(feature = "wgpu")]
pub use cubecl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "hip")]
pub use cubecl_hip as hip;

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;

#[cfg(feature = "cpu")]
pub use cubecl_cpu as cpu;

#[cfg(feature = "metal-native")]
pub use cubecl_metal as metal;

#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(test_runtime_wgpu)]
pub type TestRuntime = wgpu::WgpuRuntime;

#[cfg(test_runtime_cpu)]
pub type TestRuntime = cpu::CpuRuntime;

#[cfg(test_runtime_cuda)]
pub type TestRuntime = cuda::CudaRuntime;

#[cfg(test_runtime_hip)]
pub type TestRuntime = hip::HipRuntime;

#[cfg(test_runtime_metal)]
pub type TestRuntime = metal::MetalRuntime;
