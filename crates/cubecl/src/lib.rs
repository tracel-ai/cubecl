pub use cubecl_core::*;

pub use cubecl_runtime::config;
pub use cubecl_runtime::features;
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

#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(all(feature = "wgpu", feature = "test-runtime"))]
pub type TestRuntime = wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", feature = "test-runtime"))]
pub type TestRuntime = cpu::CpuRuntime;

#[cfg(all(feature = "cuda", feature = "test-runtime"))]
pub type TestRuntime = cuda::CudaRuntime;

#[cfg(all(feature = "hip", feature = "test-runtime"))]
pub type TestRuntime = hip::HipRuntime;
