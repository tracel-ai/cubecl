use cubecl_common::device::{Device, DeviceId};

/// The graphics API a [`WgpuDevice`] comes up on, which also settles the
/// shader compiler it is fed: SPIR-V on Vulkan, MSL on Metal, WGSL elsewhere.
///
/// [`Auto`](WgpuBackend::Auto) leaves the choice to the runtime, which tries
/// what this machine offers in the order it would want. Naming one instead
/// pins it: the device is only reachable on that API, and stays that device
/// wherever its id travels.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuBackend {
    /// Whichever of the machine's APIs the runtime would pick.
    #[default]
    Auto,
    /// Vulkan, compiling to `SPIR-V` where the build and the device allow it.
    Vulkan,
    /// Metal, compiling to MSL where the build allows it.
    Metal,
    /// `DirectX` 12.
    Dx12,
    /// `OpenGL`.
    Gl,
    /// `WebGPU`, the browser's own.
    WebGpu,
}

impl WgpuBackend {
    /// The three bits this rides in at the top of a [`DeviceId`]'s index.
    const SHIFT: u32 = 13;
    const MASK: u16 = 0x7;

    fn from_bits(bits: u16) -> Self {
        match bits {
            1 => Self::Vulkan,
            2 => Self::Metal,
            3 => Self::Dx12,
            4 => Self::Gl,
            5 => Self::WebGpu,
            _ => Self::Auto,
        }
    }
}

/// Which of a machine's wgpu devices, leaving the graphics API aside.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuDeviceKind {
    /// Discrete GPU with the given index. The index is the index of the discrete GPU in the list
    /// of all discrete GPUs found on the system.
    DiscreteGpu(usize),

    /// Integrated GPU with the given index. The index is the index of the integrated GPU in the
    /// list of all integrated GPUs found on the system.
    IntegratedGpu(usize),

    /// Virtual GPU with the given index. The index is the index of the virtual GPU in the list of
    /// all virtual GPUs found on the system.
    VirtualGpu(usize),

    /// CPU — a software rasterizer such as lavapipe, llvmpipe or WARP.
    Cpu,

    /// The best available device found with the current graphics API.
    ///
    /// This will prioritize GPUs wgpu recognizes as "high power". Additionally, you can override this using
    /// the `CUBECL_WGPU_DEFAULT_DEVICE` environment variable. This variable is spelled as if i was a `WgpuDevice`,
    /// so for example `CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(1)` or `CUBECL_WGPU_DEFAULT_DEVICE=Cpu`
    #[default]
    DefaultDevice,

    /// Use an externally created, existing, wgpu setup. This is helpful when using `CubeCL` in conjunction
    /// with some existing wgpu setup (eg. egui or bevy), as resources can be transferred in & out of `CubeCL`.
    ///
    /// The graphics API came up with the setup, so pinning one on top of this
    /// means nothing and is not encoded.
    ///
    /// # Notes
    ///
    /// This can be initialized with `init_device` in the wgpu runtime.
    Existing(u32),

    /// An adapter wgpu recognizes but does not classify — what it reports as
    /// `DeviceType::Other`, which `OpenGL` drivers often do. Indexed among
    /// its own kind like the rest.
    Other(usize),
}

impl WgpuDeviceKind {
    /// The largest index that survives a round trip through a [`DeviceId`].
    ///
    /// The graphics API rides in the top three bits of the index, leaving
    /// thirteen — far more adapters of one kind than a machine has.
    pub const MAX_INDEX: usize = (1 << WgpuBackend::SHIFT) - 1;

    fn type_id(&self) -> u16 {
        match *self {
            Self::DiscreteGpu(_) => 0,
            Self::IntegratedGpu(_) => 1,
            Self::VirtualGpu(_) => 2,
            Self::Cpu => 3,
            Self::DefaultDevice => 4,
            Self::Existing(_) => 5,
            Self::Other(_) => 6,
        }
    }

    fn index(&self) -> u16 {
        match *self {
            Self::DiscreteGpu(index)
            | Self::IntegratedGpu(index)
            | Self::VirtualGpu(index)
            | Self::Other(index) => (index & Self::MAX_INDEX) as u16,
            Self::Cpu | Self::DefaultDevice => 0,
            Self::Existing(id) => id as u16,
        }
    }
}

/// The device struct when using the `wgpu` backend.
///
/// Which device, and which graphics API to reach it on — two things a machine
/// answers separately, so they are two fields rather than one flattened list.
///
/// # Example
///
/// ```ignore
/// use cubecl_wgpu::{WgpuDevice, WgpuDeviceKind};
///
/// let first = WgpuDevice::from(WgpuDeviceKind::DiscreteGpu(0)); // First discrete GPU found.
/// let vulkan = first.clone().on(WgpuBackend::Vulkan);           // The same one, on Vulkan.
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct WgpuDevice {
    /// Which of the machine's devices.
    pub kind: WgpuDeviceKind,
    /// The graphics API to reach it on.
    pub backend: WgpuBackend,
}

impl WgpuDevice {
    /// This device on whichever graphics API the runtime would pick.
    pub fn new(kind: WgpuDeviceKind) -> Self {
        Self {
            kind,
            backend: WgpuBackend::Auto,
        }
    }

    /// The same device, pinned to `backend`.
    ///
    /// An [externally created](WgpuDeviceKind::Existing) device came up on an
    /// API of its own, so pinning one there changes nothing.
    pub fn on(self, backend: WgpuBackend) -> Self {
        match self.kind {
            WgpuDeviceKind::Existing(_) => self,
            kind => Self { kind, backend },
        }
    }
}

impl From<WgpuDeviceKind> for WgpuDevice {
    fn from(kind: WgpuDeviceKind) -> Self {
        Self::new(kind)
    }
}

impl Device for WgpuDevice {
    fn from_id(device_id: DeviceId) -> Self {
        let index = (device_id.index_id & WgpuDeviceKind::MAX_INDEX as u16) as usize;

        let kind = match device_id.type_id {
            0 => WgpuDeviceKind::DiscreteGpu(index),
            1 => WgpuDeviceKind::IntegratedGpu(index),
            2 => WgpuDeviceKind::VirtualGpu(index),
            3 => WgpuDeviceKind::Cpu,
            5 => return Self::new(WgpuDeviceKind::Existing(device_id.index_id as u32)),
            6 => WgpuDeviceKind::Other(index),
            _ => WgpuDeviceKind::DefaultDevice,
        };

        Self {
            kind,
            backend: WgpuBackend::from_bits(
                (device_id.index_id >> WgpuBackend::SHIFT) & WgpuBackend::MASK,
            ),
        }
    }

    fn to_id(&self) -> DeviceId {
        let index = self.kind.index();

        // An external setup keeps the whole index to itself: there is no API
        // to pin on it, so there are no bits to spend on one.
        let index_id = match self.kind {
            WgpuDeviceKind::Existing(_) => index,
            _ => ((self.backend as u16) << WgpuBackend::SHIFT) | index,
        };

        DeviceId::new(self.kind.type_id(), index_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both halves have to survive the id, or a device pinned to one API comes
    /// back on another.
    #[test]
    fn a_device_round_trips_its_kind_and_its_backend() {
        for kind in [
            WgpuDeviceKind::DiscreteGpu(2),
            WgpuDeviceKind::IntegratedGpu(1),
            WgpuDeviceKind::VirtualGpu(0),
            WgpuDeviceKind::Cpu,
            WgpuDeviceKind::DefaultDevice,
            WgpuDeviceKind::Other(3),
        ] {
            for backend in [
                WgpuBackend::Auto,
                WgpuBackend::Vulkan,
                WgpuBackend::Metal,
                WgpuBackend::Dx12,
                WgpuBackend::Gl,
                WgpuBackend::WebGpu,
            ] {
                let device = WgpuDevice::new(kind.clone()).on(backend);

                assert_eq!(WgpuDevice::from_id(device.to_id()), device);
            }
        }
    }

    /// The same device on two APIs must not share an id, or one client serves
    /// both and the second caller silently gets the first one's backend.
    #[test]
    fn pinning_a_backend_changes_the_id() {
        let auto = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0));

        let vulkan = auto.clone().on(WgpuBackend::Vulkan);

        assert_ne!(auto.to_id(), vulkan.to_id());
    }

    /// An external setup spends its whole index on the id it was registered
    /// with, so pinning an API on it is refused rather than silently truncating.
    #[test]
    fn an_existing_device_keeps_its_whole_index() {
        let existing = WgpuDevice::new(WgpuDeviceKind::Existing(60_000));

        let pinned = existing.clone().on(WgpuBackend::Vulkan);

        assert_eq!(pinned, existing);
        assert_eq!(WgpuDevice::from_id(existing.to_id()), existing);
    }
}
