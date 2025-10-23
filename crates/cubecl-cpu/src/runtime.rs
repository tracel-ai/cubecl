use cubecl_common::{device::DeviceState, profile::TimingMethod};
use cubecl_core::{
    CubeCount, CubeDim, MemoryConfiguration, Runtime,
    client::ComputeClient,
    ir::{StorageType, TargetProperties},
    server::ServerUtilities,
};
use cubecl_runtime::{
    DeviceProperties,
    logging::ServerLogger,
    memory_management::{
        HardwareProperties, MemoryDeviceProperties, MemoryManagement, MemoryManagementOptions,
    },
    storage::BytesStorage,
};
use cubecl_std::tensor::is_contiguous;
use sysinfo::System;

use crate::{
    compiler::{MlirCompiler, register_supported_types},
    compute::server::{CpuContext, CpuServer},
    device::CpuDevice,
};

const LOAD_WIDTH: usize = 512;

#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct CpuRuntime;

pub type CpuCompiler = MlirCompiler;

impl DeviceState for CpuServer {
    fn init(_device_id: cubecl_common::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let max_cube_dim = CubeDim::new(u32::MAX, u32::MAX, u32::MAX);
        let max_cube_count = CubeCount::Static(64, 64, 64);
        let system = System::new_all();
        let max_shared_memory_size = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory()) as usize;
        let logger = cubecl_common::stub::Arc::new(ServerLogger::default());

        let topology = HardwareProperties {
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size,
            max_cube_count,
            max_units_per_cube: u32::MAX,
            max_cube_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
        };
        let storage = BytesStorage::default();

        const ALIGNMENT: u64 = 4;
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_shared_memory_size as u64,
            alignment: ALIGNMENT,
        };

        let memory_management = MemoryManagement::from_configuration(
            storage,
            &mem_properties,
            options.memory_config,
            logger.clone(),
            MemoryManagementOptions::new("test"),
        );
        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties,
            topology,
            TimingMethod::Device,
        );
        register_supported_types(&mut device_props);

        let ctx = CpuContext::new(memory_management);
        let utilities = ServerUtilities::new(device_props, logger, ());
        CpuServer::new(ctx, utilities)
    }
}

impl Runtime for CpuRuntime {
    type Compiler = CpuCompiler;
    type Server = CpuServer;
    type Device = CpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self::Server>) -> &'static str {
        "cpu"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[64, 32, 16, 8, 4, 2, 1]
    }

    fn io_optimized_line_sizes(elem: &StorageType) -> impl Iterator<Item = u8> + Clone {
        let max = (LOAD_WIDTH / elem.size_bits()) as u8;
        let supported = Self::supported_line_sizes();
        supported.iter().filter(move |v| **v <= max).cloned()
    }

    fn io_optimized_line_sizes_unchecked(elem_size: usize) -> impl Iterator<Item = u8> + Clone {
        let elem_size_bits = elem_size * 8;
        let max = LOAD_WIDTH / elem_size_bits;
        (1..max as u8).rev().filter(|v| v.is_power_of_two())
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        is_contiguous(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }
}
