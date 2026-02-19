use cubecl_common::device::{Device, DeviceId};
use cubecl_runtime::{client::ComputeClient, runtime::Runtime};

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CubeTuneId {
    device: DeviceId,
    name: &'static str,
}

impl CubeTuneId {
    /// Create a new ID.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, device: &R::Device) -> Self {
        Self {
            device: device.to_id(),
            name: R::name(client),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "device-{}-{}-{}",
            self.device.type_id, self.device.index_id, self.name
        ))
    }
}
