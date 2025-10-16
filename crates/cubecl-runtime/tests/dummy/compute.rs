use std::sync::Arc;

use super::DummyServer;
use cubecl_common::device::{Device, DeviceState};
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryDeviceProperties, MemoryManagement, MemoryManagementOptions,
};
use cubecl_runtime::storage::BytesStorage;

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

pub type DummyClient = ComputeClient<DummyServer>;

impl DeviceState for DummyServer {
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
