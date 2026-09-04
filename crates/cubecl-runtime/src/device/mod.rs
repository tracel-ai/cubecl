//! The device type of each runtime, named without naming the runtime crate.
//!
//! Each one is a value and nothing more — which GPU, which index — so they
//! live here rather than in the runtime crates that act on them. That is what
//! lets a caller hold one, and lets `cubecl` put a single enum over them all,
//! without depending on any runtime.

mod cpu;
mod cuda;
mod hip;
mod metal;
mod wgpu;

pub use cpu::CpuDevice;
pub use cuda::CudaDevice;
pub use hip::AmdDevice;
pub use metal::MetalDevice;
pub use wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

pub use cubecl_common::device::DeviceId;
