use alloc::vec::Vec;
use cubecl_common::device::{Device, DeviceId};
use cubecl_ir::TargetProperties;
use cubecl_zspace::{Shape, Strides};

use crate::{client::Client, server::ServerStorage};

pub use crate::description::{DeviceDescription, DeviceKind};
pub use cubecl_monitoring::{
    DevicePower, DeviceTemperature, DeviceUtilization, UtilizationUnavailable,
};

/// Runtime for the `CubeCL`.
pub trait Runtime: Sized + Send + Sync + 'static + core::fmt::Debug + Clone {
    /// The compute server used to run kernels and perform autotuning.
    type Server: ServerStorage;
    /// The device used to retrieve the compute client.
    type Device: Device;

    /// Retrieve the compute client from the runtime device, initializing the
    /// server on first use.
    fn client(device: &Self::Device) -> Client {
        Client::load::<Self::Server>(device.to_id())
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

    /// The devices of `device_id`'s own kind: its peers, and itself where the
    /// machine has it.
    ///
    /// [`enumerate_devices`](Self::enumerate_devices) of its type, for a
    /// runtime whose ids carry nothing more. One whose ids do answers from the
    /// list that extra part names — a wgpu device pinned to a graphics API,
    /// from that API's, the same adapter on another being another device.
    fn enumerate_devices_like(device_id: DeviceId) -> Vec<DeviceId> {
        Self::enumerate_devices(device_id.type_id)
    }

    /// Whether this machine has the device `device_id` names — and where it
    /// does not, how many of that kind it has instead.
    ///
    /// What naming a device is checked against. The runtime's "you choose"
    /// device names no hardware of its own, so it is there as soon as the
    /// runtime found anything of its kind.
    fn find_device(device_id: DeviceId) -> Result<(), usize> {
        let of_kind = Self::enumerate_devices_like(device_id);

        let found = of_kind.contains(&device_id)
            || (device_id == Self::Device::default().to_id() && !of_kind.is_empty());

        match found {
            true => Ok(()),
            false => Err(of_kind.len()),
        }
    }

    /// Whether this machine has hardware worth choosing this runtime for.
    ///
    /// What picking a default device walks, so the question is not "could this
    /// runtime run at all" but "would a caller who did not choose want it".
    /// Enumerating a device is the answer for most runtimes; one that also
    /// exposes a software fallback answers no on a machine that has only that,
    /// leaving a native CPU runtime to claim it.
    fn is_available() -> bool {
        !Self::enumerate_all_devices().is_empty()
    }

    /// Description of the device (name, kind, memory, card, architecture, driver) read from the
    /// driver without starting a server.
    ///
    /// Returns `None` when the machine has no such device, the device is the runtime's "you
    /// choose" device (which names no hardware until a server picks some), or the runtime
    /// describes nothing without a server.
    fn description(_device: &Self::Device) -> Option<DeviceDescription> {
        None
    }

    /// How busy `device` is, by the counter its card's driver or its platform keeps. Asking
    /// creates the device's client, and must never wait on the work the device is running.
    ///
    /// A runtime that reads no counter for its devices says so.
    fn utilization(_device: &Self::Device) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Err(UtilizationUnavailable::RuntimeReadsNoCounter)
    }

    /// The current temperature of `device`. Creates a client if needed.
    ///
    /// Returns an [`UtilizationUnavailable::RuntimeReadsNoCounter`]
    /// error if the runtime doesn't read any counter.
    fn temperature(_device: &Self::Device) -> Result<DeviceTemperature, UtilizationUnavailable> {
        Err(UtilizationUnavailable::RuntimeReadsNoCounter)
    }

    /// The power usage of `device`, in Watt. Creates a client if needed.
    ///
    /// Returns an [`UtilizationUnavailable::RuntimeReadsNoCounter`]
    /// error if the runtime doesn't read any counter.
    fn power(_device: &Self::Device) -> Result<DevicePower, UtilizationUnavailable> {
        Err(UtilizationUnavailable::RuntimeReadsNoCounter)
    }
}

/// A runtime whose native stream and buffers can be handed to external
/// libraries.
///
/// Only a backend that can queue an external operation implements this, so
/// asking for [`Client::external_op`](crate::client::Client::external_op) with
/// any other runtime is a compile error. The associated types are in the
/// backend's own terms — a CUDA `CUstream` and device pointer, a HIP
/// `hipStream_t` and device pointer, a host pointer and `()` on CPU — rather
/// than a `u64` that happens to fit one of them.
pub trait NativeRuntime: Runtime {
    /// The stream external work is enqueued on: `CUstream` on CUDA,
    /// `hipStream_t` on HIP, `()` on a runtime whose external operations run on
    /// the host.
    ///
    /// Deliberately not `Send`: the stream is created and used on the device
    /// thread, and the queued callback is the only part that crosses threads.
    type NativeStream: Copy + 'static;
    /// One binding in the backend's own terms, with its size.
    type NativeBuffer: Copy + 'static;
}
