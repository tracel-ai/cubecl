use super::DummyServer;
use crate::dummy::KernelTask;
use cubecl_common::device::{Device, DeviceService};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_ir::StorageType;
use cubecl_runtime::{
    client::ComputeClient,
    compiler::{CompilationError, Compiler},
    logging::ServerLogger,
    memory_management::{MemoryConfiguration, MemoryManagement, MemoryManagementOptions},
    runtime::Runtime,
    server::ExecutionMode,
    storage::BytesStorage,
};
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

    fn client(_device: &Self::Device) -> ComputeClient<Self> {
        unimplemented!()
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        unimplemented!()
    }

    fn max_cube_count() -> (u32, u32, u32) {
        unimplemented!()
    }

    fn can_read_tensor(_shape: &[usize], _strides: &[usize]) -> bool {
        unimplemented!()
    }

    fn target_properties() -> cubecl_ir::TargetProperties {
        unimplemented!()
    }
}
