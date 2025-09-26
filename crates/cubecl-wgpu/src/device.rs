use cubecl_common::device::{Device, DeviceId};

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
    /// the `CUBECL_WGPU_DEFAULT_DEVICE` environment variable. This variable is spelled as if i was a WgpuDevice,
    /// so for example CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(1) or CUBECL_WGPU_DEFAULT_DEVICE=Cpu
    #[default]
    DefaultDevice,

    /// Deprecated, use [`DefaultDevice`](WgpuDevice::DefaultDevice).
    #[deprecated]
    BestAvailable,

    /// Use an externally created, existing, wgpu setup. This is helpful when using CubeCL in conjunction
    /// with some existing wgpu setup (eg. egui or bevy), as resources can be transferred in & out of CubeCL.
    ///
    /// # Notes
    ///
    /// This can be initialized with [`init_device`](crate::runtime::init_device).
    Existing(u32),
}

impl Device for WgpuDevice {
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => Self::DiscreteGpu(device_id.index_id as usize),
            1 => Self::IntegratedGpu(device_id.index_id as usize),
            2 => Self::VirtualGpu(device_id.index_id as usize),
            3 => Self::Cpu,
            4 => Self::DefaultDevice,
            5 => Self::Existing(device_id.index_id),
            _ => Self::DefaultDevice,
        }
    }

    fn to_id(&self) -> DeviceId {
        #[allow(deprecated)]
        match self {
            Self::DiscreteGpu(index) => DeviceId::new(0, *index as u32),
            Self::IntegratedGpu(index) => DeviceId::new(1, *index as u32),
            Self::VirtualGpu(index) => DeviceId::new(2, *index as u32),
            Self::Cpu => DeviceId::new(3, 0),
            Self::BestAvailable | WgpuDevice::DefaultDevice => DeviceId::new(4, 0),
            Self::Existing(id) => DeviceId::new(5, *id),
        }
    }

    fn device_count(type_id: u16) -> usize {
        #[cfg(target_family = "wasm")]
        {
            // WebGPU only supports a single device currently.
            1
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapters: Vec<_> = instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .filter(|adapter| {
                    // Default doesn't filter device types.
                    if type_id == 4 {
                        return true;
                    }

                    let device_type = adapter.get_info().device_type;

                    let adapter_type_id = match device_type {
                        wgpu::DeviceType::Other => 4,
                        wgpu::DeviceType::IntegratedGpu => 1,
                        wgpu::DeviceType::DiscreteGpu => 0,
                        wgpu::DeviceType::VirtualGpu => 2,
                        wgpu::DeviceType::Cpu => 3,
                    };

                    adapter_type_id == type_id
                })
                .collect();
            adapters.len()
        }
    }

    fn device_count_total() -> usize {
        #[cfg(target_family = "wasm")]
        {
            // WebGPU only supports a single device currently.
            1
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapters: Vec<_> = instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .collect();
            adapters.len()
        }
    }
}
