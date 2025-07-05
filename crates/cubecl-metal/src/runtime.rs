use crate::METAL_MEMORY_ALIGNMENT;
use crate::compute::{MetalServer, contiguous_strides};
use crate::device::MetalDevice;
use cubecl_common::benchmark::TimingMethod;
use cubecl_core::{CubeDim, Feature, Runtime};
use cubecl_cpp::metal::MslDialect;
use cubecl_cpp::metal::arch::MetalArchitecture;
use cubecl_cpp::shared::{CompilationOptions, register_wmma_features};
use cubecl_cpp::{DialectWmmaCompiler, MslCompiler};
use cubecl_runtime::channel::MutexComputeChannel;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::id::DeviceId;
use cubecl_runtime::memory_management::{
    HardwareProperties, MemoryConfiguration, MemoryDeviceProperties,
};
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::{ComputeRuntime, DeviceProperties};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Options configuring the Metal runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

impl RuntimeOptions {
    fn default() -> Self {
        Self {
            memory_config: MemoryConfiguration::default(),
        }
    }

    /// Create runtime options with custom memory configuration
    pub fn with_memory_config(memory_config: MemoryConfiguration) -> Self {
        Self { memory_config }
    }
}

#[derive(Debug)]
pub struct MetalRuntime;

type Server = MetalServer;
type Channel = MutexComputeChannel<Server>;

static RUNTIME: ComputeRuntime<MetalDevice, Server, Channel> = ComputeRuntime::new();

impl Runtime for MetalRuntime {
    type Compiler = MslCompiler;
    type Server = MetalServer;
    type Channel = MutexComputeChannel<MetalServer>;
    type Device = MetalDevice;

    fn device_id(device: &Self::Device) -> DeviceId {
        let device_hash = {
            let mut hasher = DefaultHasher::new();

            device.hash(&mut hasher);
            hasher.finish()
        };

        DeviceId::new(0, device_hash as u32)
    }

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, || create_client(device, RuntimeOptions::default()))
    }

    fn name(_client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
        "metal"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u16::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        if shape.is_empty() {
            return true;
        }

        let expected_strides = contiguous_strides(shape);
        expected_strides == strides
    }
}

fn create_client(_device: &MetalDevice, options: RuntimeOptions) -> ComputeClient<Server, Channel> {
    let metal_device = MTLCreateSystemDefaultDevice().expect("Failed to create Metal device");

    let hardware_props = {
        let max_threads_per_threadgroup = metal_device.maxThreadsPerThreadgroup().width.min(1024);
        let max_shared_memory = metal_device.maxThreadgroupMemoryLength();

        HardwareProperties {
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 31,
            max_shared_memory_size: max_shared_memory,
            max_cube_count: CubeCount::new_3d(u16::MAX as u32, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: max_threads_per_threadgroup as u32,
            max_cube_dim: CubeDim::new_3d(1024, 1024, 64),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: Some(8),
        }
    };

    let mem_properties = MemoryDeviceProperties {
        max_page_size: metal_device.recommendedMaxWorkingSetSize() / 4,
        alignment: METAL_MEMORY_ALIGNMENT as u64,
    };

    let compilation_options = CompilationOptions::default();

    let mut device_props = DeviceProperties::new(
        &[Feature::Plane],
        mem_properties.clone(),
        hardware_props,
        TimingMethod::System,
    );

    register_types(&mut device_props);

    let combinations = MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, &mut device_props);

    device_props.register_feature(Feature::SyncPlane);

    let server = MetalServer::new(
        metal_device,
        compilation_options,
        mem_properties,
        options.memory_config,
    );

    ComputeClient::new(MutexComputeChannel::new(server), device_props, ())
}

fn register_types(props: &mut DeviceProperties<Feature>) {
    use cubecl_core::ir::{Elem, FloatKind, IntKind, UIntKind};

    let mut register = |elem| {
        props.register_feature(Feature::Type(elem));
    };

    let types = [
        Elem::UInt(UIntKind::U8),
        Elem::UInt(UIntKind::U16),
        Elem::UInt(UIntKind::U32),
        Elem::UInt(UIntKind::U64),
        Elem::Int(IntKind::I8),
        Elem::Int(IntKind::I16),
        Elem::Int(IntKind::I32),
        Elem::Int(IntKind::I64),
        Elem::Float(FloatKind::F16),
        Elem::Float(FloatKind::F32),
        Elem::AtomicInt(IntKind::I32),
        Elem::AtomicUInt(UIntKind::U32),
        Elem::AtomicUInt(UIntKind::U64),
        Elem::AtomicFloat(FloatKind::F32),
        Elem::Bool,
    ];

    for ty in types {
        register(ty);
    }
}
