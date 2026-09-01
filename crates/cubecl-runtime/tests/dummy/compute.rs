use super::DummyServer;
use cubecl_common::device::{Device, DeviceService};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::server::ComputeServer;
use cubecl_runtime::{
    client::ComputeClient,
    logging::ServerLogger,
    memory_management::{MemoryConfiguration, MemoryManagement, MemoryManagementOptions},
    runtime::Runtime,
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
}

pub type DummyClient = ComputeClient<DummyRuntime>;

impl DeviceService for DummyServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        init_server(cubecl_common::device::ServiceId::of::<Self>(device_id))
    }

    fn utilities(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        ComputeServer::utilities(self) as Arc<dyn std::any::Any + Send + Sync>
    }
}

fn init_server(service: cubecl_common::device::ServiceId) -> DummyServer {
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
    DummyServer::new(service, memory_management, mem_properties)
}

pub fn test_client(device: &DummyDevice) -> DummyClient {
    ComputeClient::load(device)
}

#[derive(Debug, Clone)]
pub struct DummyRuntime;

impl Runtime for DummyRuntime {
    type Server = DummyServer;

    type Device = DummyDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn can_read_tensor(_shape: &Shape, _strides: &Strides) -> bool {
        unimplemented!()
    }

    fn target_properties() -> cubecl_ir::TargetProperties {
        unimplemented!()
    }

    fn enumerate_devices(_: u16) -> Vec<cubecl_common::device::DeviceId> {
        vec![cubecl_common::device::DeviceId {
            type_id: 0,
            index_id: 0,
        }]
    }
}
