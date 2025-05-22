use cubecl_runtime::{client::ComputeClient, id::DeviceId};

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CubeTuneId {
    device: DeviceId,
    name: &'static str,
}

impl CubeTuneId {
    /// Create a new ID.
    pub fn new<R: crate::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
    ) -> Self {
        Self {
            device: R::device_id(device),
            name: R::name(client),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "device-{}-{}-{}",
            self.device.type_id, self.device.index_id, self.name
        ))
    }
}
