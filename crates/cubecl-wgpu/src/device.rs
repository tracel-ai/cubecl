use cubecl_common::device::{Device, DeviceId, DeviceKind, DeviceRole};

/// The device struct when using the `wgpu` backend.
///
/// Note that you need to provide the device index when using a GPU backend.
///
/// # Example
///
/// ```ignore
/// use cubecl_wgpu::WgpuDevice;
///
/// let device_gpu_1 = WgpuDevice::DiscreteGpu(0); // First discrete GPU found.
/// let device_gpu_2 = WgpuDevice::DiscreteGpu(1);  // Second discrete GPU found.
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuDevice {
    /// Discrete GPU with the given index. The index is the index of the discrete GPU in the list
    /// of all discrete GPUs found on the system.
    DiscreteGpu(usize),

    /// Integrated GPU with the given index. The index is the index of the integrated GPU in the
    /// list of all integrated GPUs found on the system.
    IntegratedGpu(usize),

    /// Virtual GPU with the given index. The index is the index of the virtual GPU in the list of
    /// all virtual GPUs found on the system.
    VirtualGpu(usize),

    /// CPU.
    Cpu,

    /// The best available device found with the current [graphics API](crate::GraphicsApi).
    ///
    /// This will prioritize GPUs wgpu recognizes as "high power". Additionally, you can override this using
    /// the `CUBECL_WGPU_DEFAULT_DEVICE` environment variable. This variable is spelled as if i was a `WgpuDevice`,
    /// so for example `CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(1)` or `CUBECL_WGPU_DEFAULT_DEVICE=Cpu`
    #[default]
    DefaultDevice,

    /// Deprecated, use [`DefaultDevice`](WgpuDevice::DefaultDevice).
    #[deprecated]
    BestAvailable,

    /// Use an externally created, existing, wgpu setup. This is helpful when using `CubeCL` in conjunction
    /// with some existing wgpu setup (eg. egui or bevy), as resources can be transferred in & out of `CubeCL`.
    ///
    /// # Notes
    ///
    /// This can be initialized with [`init_device`](crate::runtime::init_device).
    Existing(u32),
}

impl Device for WgpuDevice {
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.kind {
            DeviceKind::DiscreteGpu => Self::DiscreteGpu(device_id.index_id as usize),
            DeviceKind::IntegratedGpu => Self::IntegratedGpu(device_id.index_id as usize),
            DeviceKind::VirtualGpu => Self::VirtualGpu(device_id.index_id as usize),
            DeviceKind::Cpu => Self::Cpu,
            DeviceKind::Default => Self::DefaultDevice,
        }
    }

    fn to_id(&self) -> DeviceId {
        #[allow(deprecated)]
        match self {
            Self::DiscreteGpu(index) => {
                DeviceId::new(DeviceRole::Runtime, DeviceKind::DiscreteGpu, *index as u16)
            }
            Self::IntegratedGpu(index) => DeviceId::new(
                DeviceRole::Runtime,
                DeviceKind::IntegratedGpu,
                *index as u16,
            ),
            Self::VirtualGpu(index) => {
                DeviceId::new(DeviceRole::Runtime, DeviceKind::VirtualGpu, *index as u16)
            }
            Self::Cpu => DeviceId::new(DeviceRole::Runtime, DeviceKind::Cpu, 0),
            Self::DefaultDevice | Self::BestAvailable => {
                DeviceId::new(DeviceRole::Runtime, DeviceKind::Default, 0)
            }
            Self::Existing(id) => {
                DeviceId::new(DeviceRole::Runtime, DeviceKind::DiscreteGpu, *id as u16)
            }
        }
    }
}
