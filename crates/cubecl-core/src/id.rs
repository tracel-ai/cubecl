use cubecl_common::device::{Device, DeviceId};
use cubecl_common::hash::StableHasher;
use cubecl_runtime::client::Client;

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CubeTuneId {
    device: DeviceId,
    name: &'static str,
    /// The adapter the device resolved to, which can change from run to run.
    adapter: u128,
}

impl CubeTuneId {
    /// Create a new ID.
    pub fn new(client: &Client, device: &impl Device) -> Self {
        Self {
            device: device.to_id(),
            name: client.name(),
            adapter: StableHasher::hash_one(&client.properties().identity.fingerprint),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "device-{}-{}-{}-{:x}",
            self.device.type_id, self.device.index_id, self.name, self.adapter
        ))
    }
}
