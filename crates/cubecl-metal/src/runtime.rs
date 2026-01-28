use crate::{compute::MetalServer, MetalCompiler, MetalDevice};
use cubecl_common::device::{Device, DeviceState};
use cubecl_core::{
    ir::{
        AddressType, ElemType, FloatKind, IntKind, LineSize, StorageType, TargetProperties,
        UIntKind,
    },
    Runtime,
};
use cubecl_cpp::{
    metal::{arch::MetalArchitecture, MslDialect},
    shared::register_wmma_features,
    DialectWmmaCompiler,
};
use cubecl_ir::{
    features::{EnumSet, Plane, TypeUsage},
    DeviceProperties,
};
use cubecl_runtime::client::ComputeClient;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Native Metal runtime for CubeCL
#[derive(Debug)]
pub struct MetalRuntime;

impl DeviceState for MetalServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
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
            MetalDevice::Existing(_) => {
                panic!("Existing device not yet supported");
            }
        };

        use cubecl_common::profile::TimingMethod;
        use cubecl_ir::{HardwareProperties, MemoryDeviceProperties};

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
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_props.clone(),
            hardware_props,
            TimingMethod::System,
        );

        register_metal_features(&mut device_props);

        let logger = std::sync::Arc::new(cubecl_runtime::logging::ServerLogger::default());
        let utilities = std::sync::Arc::new(cubecl_core::server::ServerUtilities::new(
            device_props.clone(),
            logger,
            (),
        ));

        let mem_config = cubecl_core::MemoryConfiguration::default();

        MetalServer::new(metal_device, mem_props.clone(), mem_config, utilities)
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

    fn supported_line_sizes() -> &'static [LineSize] {
        // Metal supports up to vec8
        &[8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(_shape: &[usize], _strides: &[usize]) -> bool {
        // Metal handles non-contiguous reads well
        true
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }
}

/// Register Metal-specific features including types, WMMA, and plane operations
fn register_metal_features(props: &mut DeviceProperties) {
    register_types(props);
    register_wmma(props);

    // Enable plane (simdgroup) operations
    props.features.alignment = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
}

/// Register supported data types for Metal
fn register_types(props: &mut DeviceProperties) {
    // Register address types
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let mut register = |elem: StorageType, usage: EnumSet<TypeUsage>| {
        props.register_type_usage(elem, usage);
    };

    // All scalar types supported by MSL
    // Note: We don't register BF16 here because Metal's simd operations don't support bfloat
    // BF16 is still usable for compute, just not for plane/simd operations
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

    // Atomic types supported by Metal
    let atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F32), // Metal 3.0+
    ];

    for ty in types {
        register(ty.into(), TypeUsage::all_scalar());
    }

    for ty in atomic_types {
        register(
            StorageType::Atomic(ty),
            TypeUsage::AtomicAdd | TypeUsage::AtomicLoadStore,
        );
    }
}

/// Register WMMA (simdgroup_matrix) features for Metal
fn register_wmma(props: &mut DeviceProperties) {
    // Get supported WMMA combinations from the MSL dialect
    let combinations = MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}
