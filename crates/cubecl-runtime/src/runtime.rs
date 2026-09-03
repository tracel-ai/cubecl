use alloc::vec::Vec;
use cubecl_common::device::{Device, DeviceId};
use cubecl_ir::TargetProperties;
use cubecl_zspace::{Shape, Strides};

use crate::{client::ComputeClient, server::ServerStorage};

/// Runtime for the `CubeCL`.
pub trait Runtime: Sized + Send + Sync + 'static + core::fmt::Debug + Clone {
    /// The compute server used to run kernels and perform autotuning.
    type Server: ServerStorage;
    /// The device used to retrieve the compute client.
    type Device: Device;

    /// Retrieve the compute client from the runtime device, initializing the
    /// server on first use.
    fn client(device: &Self::Device) -> ComputeClient {
        ComputeClient::load::<Self::Server>(device.to_id())
    }

    /// Whether a tensor with `shape` and `strides` can be read as is. If the result is false, the
    /// tensor should be made contiguous before reading.
    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool;

    /// Returns the properties of the target hardware architecture.
    fn target_properties() -> TargetProperties;

    /// Returns all devices available under the provided type id.
    fn enumerate_devices(type_id: u16) -> Vec<DeviceId>;
    /// Returns all devices that can be handled by the runtime.
    fn enumerate_all_devices() -> Vec<DeviceId> {
        Self::enumerate_devices(0)
    }
}
