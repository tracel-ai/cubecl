use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    CubeCount, CubeDim, MemoryConfiguration, Runtime,
    channel::MpscComputeChannel,
    client::ComputeClient,
    ir::{Elem, TargetProperties},
};
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties,
    id::DeviceId,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
    storage::BytesStorage,
};
use cubecl_std::tensor::is_contiguous;
use sysinfo::System;

use crate::{
    compiler::{MlirCompiler, register_supported_types},
    compute::server::{CpuContext, CpuServer},
    device::CpuDevice,
};

#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct CpuRuntime;

static RUNTIME: ComputeRuntime<CpuDevice, Server, Channel> = ComputeRuntime::new();

pub type CpuCompiler = MlirCompiler;

type Server = CpuServer;
type Channel = MpscComputeChannel<Server>;

fn create_client(options: RuntimeOptions) -> ComputeClient<Server, Channel> {
    let max_cube_dim = CubeDim::new(u32::MAX, u32::MAX, u32::MAX);
    let max_cube_count = CubeCount::Static(64, 64, 64);
    let system = System::new_all();
    let max_shared_memory_size = system
        .cgroup_limits()
        .map(|g| g.total_memory)
        .unwrap_or(system.total_memory()) as usize;

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

    let memory_management =
        MemoryManagement::from_configuration(storage, &mem_properties, options.memory_config);
    let mut device_props =
        DeviceProperties::new(&[], mem_properties, topology, TimingMethod::Device);
    register_supported_types(&mut device_props);

    let ctx = CpuContext::new(memory_management);
    let server = CpuServer::new(ctx);
    ComputeClient::new(Channel::new(server), device_props, ())
}

impl Runtime for CpuRuntime {
    type Compiler = CpuCompiler;
    type Server = CpuServer;

    type Channel = Channel;
    type Device = CpuDevice;

    fn client(_device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(_device, move || create_client(RuntimeOptions::default()))
    }

    fn name(_client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
        "cpu"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[64, 32, 16, 8, 4, 2, 1]
    }

    fn line_size_elem(elem: &Elem) -> impl Iterator<Item = u8> + Clone {
        Self::supported_line_sizes()
            .iter()
            .filter(|v| **v as usize * elem.size() <= 64)
            .cloned() // 128 bits
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn device_id(_device: &Self::Device) -> DeviceId {
        DeviceId::new(0, 0)
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        is_contiguous(shape, strides)
    }

    fn device_count() -> usize {
        1
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }
}
