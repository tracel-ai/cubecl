use crate::{
    compiler::{MlirCompiler, register_supported_types},
    compute::server::CpuServer,
    device::CpuDevice,
};
use cubecl_common::{device::DeviceState, profile::TimingMethod};
use cubecl_core::{
    CubeCount, CubeDim, MemoryConfiguration, Runtime, client::ComputeClient, ir::TargetProperties,
    server::ServerUtilities,
};
use cubecl_runtime::{
    DeviceProperties, Features,
    logging::ServerLogger,
    memory_management::{HardwareProperties, MemoryDeviceProperties},
};
use cubecl_std::tensor::is_contiguous;
use std::sync::Arc;
use sysinfo::System;

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
        let max_cube_count = CubeCount::Static(u32::MAX, u32::MAX, u32::MAX);
        let system = System::new_all();
        let max_shared_memory_size = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory()) as usize;
        let logger = cubecl_common::stub::Arc::new(ServerLogger::default());

        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();

        let topology = HardwareProperties {
            load_width: 512,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size,
            max_cube_count,
            num_cpu_cores: Some(available_parallelism as u32),
            max_units_per_cube: u32::MAX,
            max_cube_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
        };

        const ALIGNMENT: u64 = 4;
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_shared_memory_size as u64,
            alignment: ALIGNMENT,
        };

        let mut device_props = DeviceProperties::new(
            Features {
                unaligned_io: true,
                ..Default::default()
            },
            mem_properties.clone(),
            topology,
            TimingMethod::Device,
        );
        register_supported_types(&mut device_props);

        let utilities = ServerUtilities::new(device_props, logger, ());
        CpuServer::new(mem_properties, options.memory_config, Arc::new(utilities))
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

    fn supported_line_sizes() -> &'static [u8] {
        &[128, 64, 32, 16, 8, 4, 2, 1]
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
