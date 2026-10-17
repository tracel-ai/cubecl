//! The device type of each runtime, named without naming the runtime crate,
//! and the [part](crate::device::DevicePart) a device is.
//!
//! Each one is a value and nothing more: which GPU, which index, which kind of
//! card. So they live here rather than in the runtime crates that act on them,
//! which lets a caller hold one, and lets `cubecl` put a single enum over the
//! devices, without depending on any runtime.

mod cpu;
mod cuda;
mod hip;
mod metal;
mod part;
mod wgpu;

pub use cpu::CpuDevice;
pub use cuda::CudaDevice;
pub use hip::AmdDevice;
pub use metal::MetalDevice;
pub use part::DevicePart;
pub use wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

pub use cubecl_common::device::DeviceId;

/// A device index as a [`DeviceId`] carries it.
///
/// Kept at the largest the id holds rather than wrapped: wrapped, an index
/// past the end lands on a device that exists; kept, it names one no machine
/// has, and naming it is refused.
fn index_id(index: usize) -> u16 {
    index.min(u16::MAX as usize) as u16
}
