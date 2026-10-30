//! Identity and properties of a device, as its driver reports before anything runs on it.

use crate::PhysicalDevice;
use alloc::string::String;

/// Identity and properties of a device, as its driver reports before anything runs on it:
/// enough to list the machine's devices, show them to a user and pick one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    ///
    /// Never serialized, as in [`DeviceIdentity`](crate::DeviceIdentity): every id on a card is
    /// local to the host that reported it, so a description read on another host would match that
    /// host's own card at the same PCI address.
    #[cfg_attr(feature = "serde", serde(skip))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;

    /// A description travels whole but for the card, whose ids only mean something on the host
    /// that read them.
    #[test]
    fn a_description_travels_without_its_card() {
        let mut description =
            DeviceDescription::new("AMD Radeon 8060S Graphics", DeviceKind::Integrated);
        description.max_memory = Some(64 << 30);
        description.architecture = Some("gfx1151".into());
        description.physical = Some(PhysicalDevice::default());

        let read: DeviceDescription =
            serde_json::from_str(&serde_json::to_string(&description).unwrap()).unwrap();

        assert_eq!(read.physical, None);
        assert_eq!(
            read,
            DeviceDescription {
                physical: None,
                ..description
            }
        );
    }
}
