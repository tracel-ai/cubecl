use crate::{compute::affinity, compute::server::CpuServer, device::CpuDevice};
use cubecl_common::{device::DeviceService, profile::TimingMethod};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    client::ComputeClient,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        AddressType, DeviceIdentity, DeviceProperties, ElemType, FloatKind, HardwareProperties,
        IntKind, MemoryDeviceProperties, TargetProperties, Type, UIntKind, VectorSize,
        features::{AtomicUsage, Features, TypeUsage},
    },
    server::ServerUtilities,
    zspace::{Shape, Strides},
};
use cubecl_llvm::PlironCompiler;
use cubecl_runtime::{allocator::ContiguousMemoryLayoutPolicy, logging::ServerLogger};
use cubecl_std::tensor::is_contiguous;
use std::sync::Arc;
use sysinfo::System;

#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct CpuRuntime;

pub type CpuCompiler = PlironCompiler;

fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let supported_types = [
        ElemType::Index,
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
        ElemType::Float(FloatKind::F64),
        ElemType::Bool,
    ];

    let supported_atomic_types = [
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F64),
        ElemType::Bool,
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in [FloatKind::E4M3, FloatKind::E5M2] {
        props.register_type_usage(
            ElemType::Float(ty),
            TypeUsage::Conversion | TypeUsage::Buffer,
        );
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
    }
}

impl DeviceService for CpuServer {
    fn init(_device_id: cubecl_common::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let mut system = System::new();
        system.refresh_memory();
        // Bounds the allocator's page size, not a kernel's shared memory.
        let total_memory = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory()) as usize;
        let logger = cubecl_environment::sync::Arc::new(ServerLogger::default());

        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();
        let available_parallelism = available_parallelism as u32;
        let max_cube_dim = (
            available_parallelism,
            available_parallelism,
            available_parallelism,
        );
        let max_cube_count = (u32::MAX, u32::MAX, u32::MAX);
        // Kernels size their stages against shared memory ("as big as shared
        // memory allows"), so reporting whole RAM lets one matmul launch
        // reserve tens of GB and abort. GPU shared memory is carved from L1,
        // so the L1d size is its honest CPU analogue — reporting the L2
        // measured ~2.5x worse on decode gemv, stages outgrowing what stays
        // resident. GPU-like floor when the topology cannot be read.
        let max_shared_memory_size = affinity::l1d_cache_size().unwrap_or(64 * 1024);
        let topology = HardwareProperties {
            load_width: 512,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size,
            max_cube_count,
            num_cpu_cores: Some(available_parallelism as u32),
            last_level_cache_size: affinity::llc_cache_size(),
            max_units_per_cube: available_parallelism,
            max_cube_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };

        const ALIGNMENT: u64 = 8;

        let mem_properties = MemoryDeviceProperties {
            max_page_size: total_memory as u64,
            alignment: ALIGNMENT,
        };

        let mut device_props = DeviceProperties::new(
            Features {
                unaligned_io: true,
                ..Default::default()
            },
            mem_properties.clone(),
            topology.clone(),
            TimingMethod::Device,
            // The CPU backend JITs through LLVM for whatever host it runs on
            // and persists no compiled code, so there is no per-machine
            // namespace to match against. The architecture is the honest
            // fingerprint: it is what the generated code is valid for.
            DeviceIdentity {
                name: "CPU".to_string(),
                fingerprint: format!("cpu_{}", std::env::consts::ARCH),
            },
        );
        register_supported_types(&mut device_props);

        let utilities = ServerUtilities::new(
            device_props,
            logger,
            (),
            ContiguousMemoryLayoutPolicy::new(ALIGNMENT as usize),
        );
        CpuServer::new(mem_properties, options.memory_config, Arc::new(utilities))
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

impl Runtime for CpuRuntime {
    type Compiler = CpuCompiler;
    type Server = CpuServer;
    type Device = CpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "cpu"
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        is_contiguous(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as cubecl_core::server::ComputeServer>::Info,
    ) -> Vec<DeviceId> {
        vec![DeviceId {
            type_id: 0,
            index_id: 0,
        }]
    }
}
