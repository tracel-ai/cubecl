pub use cubecl_core::*;

use cubecl_core::client::ComputeClient;

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

/// A device of any runtime this build enables.
///
/// Each runtime has its own device type, and until now the only way to reach a
/// [`ComputeClient`] was through the runtime, `R::client(&device)`. This enum
/// is the runtime-independent way: a value names both the runtime and the
/// device on it, and [`client`](Device::client) hands back the client for it.
///
/// The variants follow the runtime features, so a match on this type needs a
/// wildcard arm.
///
/// ```no_run
/// # #[cfg(feature = "cuda")]
/// # fn main() {
/// use cubecl::Device;
///
/// let device = Device::Cuda(cubecl::cuda::CudaDevice::new(0));
/// let client = device.client();
/// println!("running on {}", client.name());
/// # }
/// # #[cfg(not(feature = "cuda"))]
/// # fn main() {}
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// A device of the CUDA runtime.
    #[cfg(feature = "cuda")]
    Cuda(cubecl_cuda::CudaDevice),
    /// A device of the HIP runtime.
    #[cfg(feature = "hip")]
    Hip(cubecl_hip::AmdDevice),
    /// A device of the native Metal runtime.
    #[cfg(feature = "metal-native")]
    Metal(cubecl_metal::MetalDevice),
    /// A device of the wgpu runtime, on its default compiler.
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl_wgpu::WgpuDevice),
    /// The device of the CPU runtime.
    #[cfg(feature = "cpu")]
    Cpu(cubecl_cpu::CpuDevice),
}

impl Device {
    /// The compute client of this device, initialized on first use.
    pub fn client(&self) -> ComputeClient {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ref device) => cubecl_cuda::CudaRuntime::client(device),
            #[cfg(feature = "hip")]
            Self::Hip(ref device) => cubecl_hip::HipRuntime::client(device),
            #[cfg(feature = "metal-native")]
            Self::Metal(ref device) => cubecl_metal::MetalRuntime::client(device),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(ref device) => <cubecl_wgpu::WgpuRuntime>::client(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(ref device) => cubecl_cpu::CpuRuntime::client(device),
        }
    }
}

#[cfg(feature = "cuda")]
impl From<cubecl_cuda::CudaDevice> for Device {
    fn from(device: cubecl_cuda::CudaDevice) -> Self {
        Self::Cuda(device)
    }
}

#[cfg(feature = "hip")]
impl From<cubecl_hip::AmdDevice> for Device {
    fn from(device: cubecl_hip::AmdDevice) -> Self {
        Self::Hip(device)
    }
}

#[cfg(feature = "metal-native")]
impl From<cubecl_metal::MetalDevice> for Device {
    fn from(device: cubecl_metal::MetalDevice) -> Self {
        Self::Metal(device)
    }
}

#[cfg(feature = "wgpu")]
impl From<cubecl_wgpu::WgpuDevice> for Device {
    fn from(device: cubecl_wgpu::WgpuDevice) -> Self {
        Self::Wgpu(device)
    }
}

#[cfg(feature = "cpu")]
impl From<cubecl_cpu::CpuDevice> for Device {
    fn from(device: cubecl_cpu::CpuDevice) -> Self {
        Self::Cpu(device)
    }
}

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
