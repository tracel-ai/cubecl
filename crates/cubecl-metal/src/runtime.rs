use crate::{MetalDevice, compute::MetalServer};
use cubecl_common::device::{Device, DeviceService};
use cubecl_core::{
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        DeviceIdentity, DeviceProperties, HardwareProperties, MemoryDeviceProperties,
        PhysicalDevice, RegistryEntryId, TargetProperties,
    },
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::metal::mma::mma_properties_metal;
use cubecl_monitoring::{
    DevicePower, DeviceTemperature, DeviceUtilization, UtilizationUnavailable,
    gpu_utilization::CardCounters,
};
use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_server::runtime::{DeviceDescription, DeviceKind, Runtime};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLGPUFamily};

/// Native Metal runtime for `CubeCL`.
#[derive(Debug, Clone)]
pub struct MetalRuntime;

impl DeviceService for MetalServer {
    fn init(device_id: DeviceId) -> Self {
        let device = MetalDevice::from_id(device_id);
        let metal_device = resolve(&device).unwrap_or_else(|missing| panic!("{missing}"));

        ensure_metal3(&metal_device);

        use cubecl_common::profile::TimingMethod;

        // Metal states no total, only the working set it recommends a device
        // stay under before it starts paging, around two thirds of unified
        // memory on Apple silicon. That is the budget `max_memory` asks for.
        let mem_props = MemoryDeviceProperties::new((*metal_device).maxBufferLength() as u64, 256)
            .with_max_memory((*metal_device).recommendedMaxWorkingSetSize());

        let hardware_props = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            // Metal allows 31 buffer bindings; one is reserved for the per-kernel info buffer.
            max_bindings: 30,
            max_shared_memory_size: (*metal_device).maxThreadgroupMemoryLength(),
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: (*metal_device).maxThreadsPerThreadgroup().width as u32,
            max_cube_dim: {
                let size = (*metal_device).maxThreadsPerThreadgroup();
                (size.width as u32, size.height as u32, size.depth as u32)
            },
            // Metal reports no core count; the GPU's driver states it in the I/O Registry.
            #[cfg(target_os = "macos")]
            num_streaming_multiprocessors: cubecl_monitoring::gpu_utilization::gpu_core_count(
                RegistryEntryId::new(metal_device.registryID()),
            ),
            #[cfg(not(target_os = "macos"))]
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: 4,
            cube_mma_reserved_shared_memory: 0,
        };

        // Metal is the one backend where the display name *is* the fingerprint:
        // the MSL is emitted from device-derived compilation options, so the
        // device's own name is what keeps sources built for another GPU out.
        // `MetalPipelines` builds the same `msl_{name}` namespace from the same
        // string.
        let device_name = metal_device.name().to_string();
        let physical = physical_device(&metal_device);

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_props.clone(),
            hardware_props,
            TimingMethod::Device,
            DeviceIdentity {
                fingerprint: format!("msl_{device_name}"),
                name: device_name,
                physical: Some(physical),
            },
        );

        cubecl_cpp::metal::register_metal_features(&mut device_props);

        let logger = std::sync::Arc::new(cubecl_server::logging::ServerLogger::default());
        let allocator = ContiguousMemoryLayoutPolicy::new(mem_props.alignment as usize);
        let (utilities, captures) = cubecl_core::server::ServerUtilities::init(
            cubecl_common::device::ServiceId::of::<Self>(device_id),
            "metal",
            device_props.clone(),
            MetalRuntime::target_properties(),
            logger,
            allocator,
        );
        let utilities = std::sync::Arc::new(utilities);

        let mem_config = cubecl_core::MemoryConfiguration::default();

        MetalServer::new(
            metal_device,
            mem_props.clone(),
            mem_config,
            utilities,
            captures,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone() as ServerUtilitiesHandle
    }
}

impl Runtime for MetalRuntime {
    type Server = MetalServer;
    type Device = MetalDevice;

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: mma_properties_metal(),
        }
    }

    fn enumerate_devices(type_id: u16) -> Vec<DeviceId> {
        // type_id matches `device.rs`: 0=Default, 1=Discrete, 2=Integrated, where
        // integrated == `hasUnifiedMemory()`. index_id is the nth device of that class,
        // as resolved by `DeviceService::init`.
        match type_id {
            0 => vec![DeviceId {
                type_id: 0,
                index_id: 0,
            }],
            1 => crate::device::all_devices()
                .into_iter()
                .filter(|d| !(**d).hasUnifiedMemory())
                .enumerate()
                .map(|(idx, _)| DeviceId {
                    type_id: 1,
                    index_id: idx as u16,
                })
                .collect(),
            2 => crate::device::all_devices()
                .into_iter()
                .filter(|d| (**d).hasUnifiedMemory())
                .enumerate()
                .map(|(idx, _)| DeviceId {
                    type_id: 2,
                    index_id: idx as u16,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    fn utilization(device: &Self::Device) -> Result<DeviceUtilization, UtilizationUnavailable> {
        CardCounters::utilization(Self::client(device).properties().identity.physical.as_ref())
    }

    fn temperature(device: &Self::Device) -> Result<DeviceTemperature, UtilizationUnavailable> {
        CardCounters::temperature(Self::client(device).properties().identity.physical.as_ref())
    }

    fn power(device: &Self::Device) -> Result<DevicePower, UtilizationUnavailable> {
        CardCounters::power(Self::client(device).properties().identity.physical.as_ref())
    }

    fn description(device: &Self::Device) -> Option<DeviceDescription> {
        let metal_device = resolve(device).ok()?;
        let kind = match (*metal_device).hasUnifiedMemory() {
            true => DeviceKind::Integrated,
            false => DeviceKind::Discrete,
        };
        let mut description = DeviceDescription::new(metal_device.name().to_string(), kind);
        // Metal states no total, only the working set it recommends staying under.
        description.max_memory = Some((*metal_device).recommendedMaxWorkingSetSize());
        description.physical = Some(physical_device(&metal_device));
        Some(description)
    }
}

/// The card behind `metal_device`, found by its I/O Registry entry.
fn physical_device(metal_device: &ProtocolObject<dyn MTLDevice>) -> PhysicalDevice {
    let mut physical = PhysicalDevice::default();
    physical.registry_entry_id = Some(RegistryEntryId::new(metal_device.registryID()));
    physical
}

/// The Metal device `device` reports, or why it does not find one: the first
/// of its class, discrete or integrated (`hasUnifiedMemory`), counted as
/// `enumerate_devices` counts them.
fn resolve(
    device: &MetalDevice,
) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>, &'static str> {
    let nth_of_class = |unified: bool, index: usize| {
        crate::device::all_devices()
            .into_iter()
            .filter(|metal_device| (**metal_device).hasUnifiedMemory() == unified)
            .nth(index)
    };
    match *device {
        MetalDevice::DefaultDevice => {
            crate::device::default_device().ok_or("No Metal device found")
        }
        MetalDevice::DiscreteGpu(index) => {
            nth_of_class(false, index).ok_or("Discrete GPU not found")
        }
        MetalDevice::IntegratedGpu(index) => {
            nth_of_class(true, index).ok_or("Integrated GPU not found")
        }
        MetalDevice::Existing(id) => crate::device::get_existing_device(id)
            .ok_or("Existing device not found. Use register_device() first."),
    }
}

fn ensure_metal3(device: &ProtocolObject<dyn MTLDevice>) {
    if !device.supportsFamily(MTLGPUFamily::Metal3) {
        let name = device.name().to_string();
        panic!(
            "CubeCL Metal backend requires Metal 3.0+. Device '{name}' does not support Metal 3."
        );
    }
}
