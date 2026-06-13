pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

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

#[cfg(feature = "metal")]
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
