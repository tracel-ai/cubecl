//! `CubeCL`: the language, the launch API, and the runtimes.
//!
//! A kernel library depends on this crate with no runtime feature on and
//! compiles against no runtime at all. A binary, a benchmark or a test suite
//! turns on the features for the runtimes it wants to link, and then a
//! [`Device`] hands back the [`Client`](client::Client) to launch on —
//! [`Device::default`] for the most capable runtime in the build, or a named
//! variant for a particular one.
//!
//! ```no_run
//! # #[cfg(feature = "wgpu")]
//! # fn main() {
//! let client = cubecl::Device::default().client();
//! # }
//! # #[cfg(not(feature = "wgpu"))]
//! # fn main() {}
//! ```

extern crate alloc;

pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

/// Ship pre-warmed autotune and compilation caches with an application.
///
/// Run the application once so its caches are warm, then save the active
/// environment to a file and ship it. On the target machine either
/// [`environment::load`] mounts the file in place, or [`import`] copies it
/// into the local environment, after which the file can be deleted.
///
/// The exporting binary must be built against the same cubecl version as the
/// consuming one. The version is part of every cache namespace, so a bundle
/// built elsewhere installs cleanly and is then ignored, with a warning as the
/// only signal. Calling this from your own crate is what keeps the two in
/// step.
///
/// ```no_run
/// use cubecl::bundle::BundleFormat;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // ... run the work you want tuned and compiled, then:
/// let manifest = cubecl::environment::bundle().save("h100.bundle", BundleFormat::Sqlite)?;
/// println!("exported {}", manifest.name);
/// # Ok(())
/// # }
/// ```
///
/// Merging several roots or restricting the namespaces goes through
/// [`export`] directly.
///
/// [`export`]: cubecl_environment::bundle::export
/// [`import`]: cubecl_environment::bundle::import
/// [`environment::load`]: cubecl_environment::environment::load
pub use cubecl_environment::bundle;

/// Which named environment caches are warmed into, and where it lives.
pub use cubecl_environment::environment;

/// Running a workload for the compilation and tuning it provokes, without
/// running the workload itself.
///
/// This is what makes producing a [`bundle`] affordable: inside a
/// [`DryRun`](dry_run::DryRun) every launch is compiled, cached and tuned
/// without also being executed. Buffers are left as they were, so it only suits
/// a pass driven by the *shapes* it produces.
///
/// ```no_run
/// # fn warm_up() {}
/// let _dry_run = cubecl::dry_run::DryRun::new();
/// warm_up();
/// ```
pub use cubecl_runtime::dry_run;

/// Watching what the runtime runs: the launch observer, and the profiling
/// logger's levels.
///
/// Re-exported because a caller attributing kernels to its own work reaches for
/// [`LaunchObservation`](cubecl_runtime::logging::LaunchObservation) and has no
/// other reason to depend on `cubecl-runtime` directly.
pub use cubecl_runtime::logging;

/// A device of any runtime, and the runtime it belongs to.
pub mod device;

pub use device::{Device, RuntimeId};

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;

#[cfg(feature = "wgpu")]
pub use cubecl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "hip")]
pub use cubecl_hip as hip;

#[cfg(feature = "cpu")]
pub use cubecl_cpu as cpu;

#[cfg(feature = "metal-native")]
pub use cubecl_metal as metal;

/// The runtime this build tests on.
#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_wgpu)]
pub type TestRuntime = wgpu::WgpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_cpu)]
pub type TestRuntime = cpu::CpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_cuda)]
pub type TestRuntime = cuda::CudaRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_hip)]
pub type TestRuntime = hip::HipRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_metal)]
pub type TestRuntime = metal::MetalRuntime;

/// The client of [`test_device`], for the test that only wants one.
///
/// ```no_run
/// let client = cubecl::test_client();
/// ```
#[cfg(any(
    test_runtime_default,
    test_runtime_wgpu,
    test_runtime_cpu,
    test_runtime_cuda,
    test_runtime_hip,
    test_runtime_metal
))]
pub fn test_client() -> client::Client {
    test_device().client()
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
///
/// A test reaches its client through [`Device::client`] like any other caller,
/// which is what keeps it from naming a runtime to get one.
///
/// ```no_run
/// let client = cubecl::test_device().client();
/// ```
#[cfg(any(test_runtime_default, test_runtime_wgpu))]
pub fn test_device() -> Device {
    Device::Wgpu(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_cpu)]
pub fn test_device() -> Device {
    Device::Cpu(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_cuda)]
pub fn test_device() -> Device {
    Device::Cuda(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_hip)]
pub fn test_device() -> Device {
    Device::Hip(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_metal)]
pub fn test_device() -> Device {
    Device::Metal(Default::default())
}
