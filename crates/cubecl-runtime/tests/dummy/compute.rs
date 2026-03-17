use super::DummyServer;
use crate::dummy::KernelTask;
use cubecl_common::device::{Device, DeviceService};
use cubecl_common::profile::TimingMethod;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_ir::StorageType;
use cubecl_ir::{DeviceProperties, HardwareProperties};
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::server::ComputeServer;
use cubecl_runtime::server::ServerUtilities;
use cubecl_runtime::{
    client::ComputeClient,
    compiler::{CompilationError, Compiler},
    logging::ServerLogger,
    memory_management::{MemoryConfiguration, MemoryManagement, MemoryManagementOptions},
    runtime::Runtime,
    server::ExecutionMode,
    storage::BytesStorage,
};
use cubecl_zspace::Shape;
use cubecl_zspace::Strides;
use std::sync::Arc;

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct DummyDevice;

impl Device for DummyDevice {
    fn from_id(_device_id: cubecl_common::device::DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> cubecl_common::device::DeviceId {
        cubecl_common::device::DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

pub type DummyClient = ComputeClient<DummyRuntime>;

impl DeviceService for DummyServer {
    fn init(_device_id: cubecl_common::device::DeviceId) -> Self {
        init_server()
    }

    fn utilities(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        Arc::new(dummy_utilities::<Self>())
    }
}

fn init_server() -> DummyServer {
    let storage = BytesStorage::default();
    let mem_properties = MemoryDeviceProperties {
        max_page_size: 1024 * 1024 * 512,
        alignment: 32,
    };

    let memory_management = MemoryManagement::from_configuration(
        storage,
        &mem_properties,
        MemoryConfiguration::default(),
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main CPU Memory"),
    );
    DummyServer::new(memory_management, mem_properties)
}

pub fn test_client(device: &DummyDevice) -> DummyClient {
    ComputeClient::load(device)
}

fn dummy_utilities<
    S: ComputeServer<Info = (), MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy>,
>() -> ServerUtilities<S> {
    ServerUtilities::new(
        DeviceProperties::new(
            Default::default(),
            MemoryDeviceProperties {
                max_page_size: 1,
                alignment: 1,
            },
            HardwareProperties {
                load_width: 1,
                plane_size_min: 1,
                plane_size_max: 1,
                max_bindings: 1,
                max_shared_memory_size: 1,
                max_cube_count: (1, 1, 1),
                max_units_per_cube: 1,
                max_cube_dim: (1, 1, 1),
                num_streaming_multiprocessors: Some(1),
                num_tensor_cores: Some(1),
                min_tensor_cores_dim: None,
                num_cpu_cores: None,
                max_vector_size: 1,
            },
            TimingMethod::System,
        ),
        Arc::new(ServerLogger::default()),
        (),
        ContiguousMemoryLayoutPolicy::new(0),
    )
}

#[derive(Debug, Clone)]
pub struct DummyCompiler;

impl Compiler for DummyCompiler {
    type Representation = KernelTask;

    type CompilationOptions = ();

    fn compile(
        &mut self,
        _kernel: cubecl_runtime::kernel::KernelDefinition,
        _compilation_options: &Self::CompilationOptions,
        _mode: ExecutionMode,
        _addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        unimplemented!()
    }

    fn elem_size(&self, _elem: cubecl_ir::ElemType) -> usize {
        unimplemented!()
    }

    fn extension(&self) -> &'static str {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct DummyRuntime;

impl Runtime for DummyRuntime {
    type Compiler = DummyCompiler;

    type Server = DummyServer;

    type Device = DummyDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        unimplemented!()
    }

    fn max_cube_count() -> (u32, u32, u32) {
        unimplemented!()
    }

    fn can_read_tensor(_shape: &Shape, _strides: &Strides) -> bool {
        unimplemented!()
    }

    fn target_properties() -> cubecl_ir::TargetProperties {
        unimplemented!()
    }
}
