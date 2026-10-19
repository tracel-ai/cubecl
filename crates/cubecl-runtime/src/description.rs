//! Identity and properties of a device, as its driver reports before anything runs on it.

use alloc::string::String;
use cubecl_ir::PhysicalDevice;

/// Identity and properties of a device, as its driver reports before anything runs on it:
/// enough to list the machine's devices, show them to a user and pick one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceDescription {
    /// The name the device reports, e.g. `NVIDIA GeForce RTX 5070 Laptop GPU`.
    pub name: String,
    /// The kind of device.
    pub kind: DeviceKind,
    /// The memory the device can use, in bytes. The card's own memory for a GPU, the machine's RAM
    /// for the CPU runtime, and on Metal the working set the device recommends staying under.
    /// `None` where the runtime's API does not say.
    pub max_memory: Option<u64>,
    /// The card the device runs on. `None` for a device that is not a card: a CPU or a software
    /// rasterizer.
    pub physical: Option<PhysicalDevice>,
    /// The architecture the runtime compiles for: the NVPTX target on CUDA, the
    /// `gcnArchName` on HIP, the LLVM CPU name on the CPU runtime.
    /// `None` on Metal and wgpu.
    pub architecture: Option<String>,
    /// The driver's version, in each API's own terms, so two runtimes on one card can disagree:
    /// on CUDA the CUDA version the driver supports (`13.0`), on HIP the HIP version
    /// (`7.1.25424`), on wgpu the driver's name and release (`NVIDIA 580.82.09`). `None` on Metal
    /// and the CPU runtime.
    pub driver_version: Option<String>,
}

impl DeviceDescription {
    /// A device named `name`, of `kind`.
    pub fn new(name: impl Into<String>, kind: DeviceKind) -> Self {
        Self {
            name: name.into(),
            kind,
            max_memory: None,
            physical: None,
            architecture: None,
            driver_version: None,
        }
    }
}

/// What sort of device a [`DeviceDescription`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// A GPU with memory of its own.
    Discrete,
    /// A GPU sharing the host's memory: an APU, Apple silicon, an Intel iGPU.
    Integrated,
    /// A GPU a hypervisor hands to a virtual machine.
    Virtual,
    /// The host's processors, or a software rasterizer running on them.
    Cpu,
    /// Whatever the driver calls none of the above.
    Other,
}
