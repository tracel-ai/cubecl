use crate::{MetalCompiler, MetalDevice, compute::MetalServer};
use cubecl_common::device::{Device, DeviceService};
use cubecl_core::{
    Runtime,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        AddressType, DeviceProperties, ElemType, FloatKind, HardwareProperties, IntKind,
        MemoryDeviceProperties, StorageType, TargetProperties, Type, UIntKind,
        features::{AtomicUsage, Plane, TypeUsage},
    },
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    DialectWmmaCompiler,
    metal::{MslDialect, arch::MetalArchitecture},
    shared::register_wmma_features,
};
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::client::ComputeClient;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLGPUFamily};

/// Native Metal runtime for `CubeCL`.
#[derive(Debug, Clone)]
pub struct MetalRuntime;

impl DeviceService for MetalServer {
    fn init(device_id: DeviceId) -> Self {
        let device = MetalDevice::from_id(device_id);
        let metal_device = match device {
            MetalDevice::DefaultDevice => {
                crate::device::default_device().expect("No Metal device found")
            }
            MetalDevice::DiscreteGpu(idx) => {
                let devices = crate::device::all_devices();
                devices
                    .into_iter()
                    .filter(
                        |d: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>| {
                            !(**d).isLowPower()
                        },
                    )
                    .nth(idx)
                    .expect("Discrete GPU not found")
            }
            MetalDevice::IntegratedGpu(idx) => {
                let devices = crate::device::all_devices();
                devices
                    .into_iter()
                    .filter(
                        |d: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>| {
                            (**d).isLowPower()
                        },
                    )
                    .nth(idx)
                    .expect("Integrated GPU not found")
            }
            MetalDevice::Existing(id) => crate::device::get_existing_device(id)
                .expect("Existing device not found. Use register_device() first."),
        };

        ensure_metal3(&metal_device);

        use cubecl_common::profile::TimingMethod;

        let mem_props = MemoryDeviceProperties {
            max_page_size: (*metal_device).maxBufferLength() as u64,
            alignment: 256,
        };

        let hardware_props = HardwareProperties {
            load_width: 128,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 31,
            max_shared_memory_size: (*metal_device).maxThreadgroupMemoryLength() as usize,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: (*metal_device).maxThreadsPerThreadgroup().width as u32,
            max_cube_dim: {
                let size = (*metal_device).maxThreadsPerThreadgroup();
                (size.width as u32, size.height as u32, size.depth as u32)
            },
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            max_vector_size: 4,
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_props.clone(),
            hardware_props,
            TimingMethod::System,
        );

        register_metal_features(&mut device_props);

        let logger = std::sync::Arc::new(cubecl_runtime::logging::ServerLogger::default());
        let allocator = ContiguousMemoryLayoutPolicy::new(mem_props.alignment as usize);
        let utilities = std::sync::Arc::new(cubecl_core::server::ServerUtilities::new(
            device_props.clone(),
            logger,
            (),
            allocator,
        ));

        let mem_config = cubecl_core::MemoryConfiguration::default();

        MetalServer::new(metal_device, mem_props.clone(), mem_config, utilities)
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone() as ServerUtilitiesHandle
    }
}

impl Runtime for MetalRuntime {
    type Compiler = MetalCompiler;
    type Server = MetalServer;
    type Device = MetalDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "metal"
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }

    fn enumerate_devices(
        _type_id: u16,
        _info: &<Self::Server as cubecl_core::server::ComputeServer>::Info,
    ) -> Vec<DeviceId> {
        let devices = crate::device::all_devices();
        (0..devices.len())
            .map(|i| DeviceId {
                type_id: 0,
                index_id: i as u32,
            })
            .collect()
    }
}

/// Register Metal-specific features including types, WMMA, and plane operations
fn register_metal_features(props: &mut DeviceProperties) {
    register_types(props);
    register_wmma(props);

    props.features.alignment = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
    props.features.plane.insert(Plane::NonUniformControlFlow);
}

/// Register supported data types for Metal
fn register_types(props: &mut DeviceProperties) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let types = [
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    let atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F32),
    ];

    for ty in types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in atomic_types {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ty)),
            AtomicUsage::Add | AtomicUsage::LoadStore,
        );
    }
}

/// Register WMMA (`simdgroup_matrix`) features for Metal.
fn register_wmma(props: &mut DeviceProperties) {
    let combinations = MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}

fn ensure_metal3(device: &ProtocolObject<dyn MTLDevice>) {
    if !device.supportsFamily(MTLGPUFamily::Metal3) {
        let name = device.name().to_string();
        panic!(
            "CubeCL Metal backend requires Metal 3.0+. Device '{name}' does not support Metal 3."
        );
    }
}
