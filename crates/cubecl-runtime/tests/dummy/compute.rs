use super::DummyServer;
use cubecl_runtime::channel::MutexComputeChannel;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryDeviceProperties, MemoryManagement,
};
use cubecl_runtime::storage::BytesStorage;
use cubecl_runtime::tune::{AutotuneOperationSet, LocalTuner};
use cubecl_runtime::{ComputeRuntime, DeviceProperties};

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyDevice;

pub type DummyChannel = MutexComputeChannel<DummyServer>;
pub type DummyClient = ComputeClient<DummyServer, DummyChannel>;

static RUNTIME: ComputeRuntime<DummyDevice, DummyServer, DummyChannel> = ComputeRuntime::new();
pub static TUNER_DEVICE_ID: &str = "dummy-device";
pub static TUNER_PREFIX: &str = "dummy-tests";
pub static TEST_TUNER: LocalTuner<String, String> = LocalTuner::new(TUNER_PREFIX);

pub fn autotune_execute(
    client: &ComputeClient<DummyServer, MutexComputeChannel<DummyServer>>,
    set: Box<dyn AutotuneOperationSet<String>>,
) {
    TEST_TUNER.execute(&TUNER_DEVICE_ID.to_string(), client, set)
}

pub fn init_client() -> ComputeClient<DummyServer, MutexComputeChannel<DummyServer>> {
    let storage = BytesStorage::default();
    let mem_properties = MemoryDeviceProperties {
        max_page_size: 1024 * 1024 * 512,
        alignment: 32,
    };
    let memory_management = MemoryManagement::from_configuration(
        storage,
        mem_properties.clone(),
        MemoryConfiguration::default(),
    );
    let server = DummyServer::new(memory_management);
    let channel = MutexComputeChannel::new(server);
    ComputeClient::new(channel, DeviceProperties::new(&[], mem_properties))
}

pub fn client(device: &DummyDevice) -> DummyClient {
    RUNTIME.client(device, init_client)
}
