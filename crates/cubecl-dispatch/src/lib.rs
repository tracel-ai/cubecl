#![warn(missing_docs)]

//! Runtime selection for `CubeCL`.
//!
//! A kernel library depends on `cubecl` and never on a runtime. A binary, a
//! benchmark or a test suite depends on this crate with the runtime features it
//! wants, and gets three things: the runtime crates themselves, the runtime
//! tests run on, and the one thing `cubecl` cannot do on its own, which is to
//! turn a [`Device`](cubecl::Device) into a [`Client`] — [`DeviceExt::client`].
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # fn main() {
//! use cubecl::Device;
//! use cubecl_dispatch::DeviceExt;
//!
//! let device = Device::Cuda(cubecl_dispatch::cuda::CudaDevice::new(0));
//! let client = device.client();
//! println!("running on {}", client.name());
//! # }
//! # #[cfg(not(feature = "cuda"))]
//! # fn main() {}
//! ```

extern crate alloc;

#[cfg(any_runtime)]
use cubecl::Device;
#[cfg(any_runtime)]
use cubecl::RuntimeId;
use cubecl::client::Client;
#[cfg(any_runtime)]
use cubecl::device::{Device as DeviceIdentity, DeviceId};
#[cfg(any_runtime)]
use cubecl::zspace::{Shape, Strides};
#[cfg(any_runtime)]
use cubecl_runtime::runtime::Runtime;

#[cfg(feature = "wgpu")]
pub use cubecl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "hip")]
pub use cubecl_hip as hip;

#[cfg(feature = "cpu")]
pub use cubecl_cpu as cpu;

#[cfg(feature = "metal-native")]
pub use cubecl_metal as metal;

/// What a [`Device`](cubecl::Device) can do once the runtimes are linked.
///
/// `cubecl` names a device without naming its runtime, so answering for one
/// takes the runtime crates, which are this crate's to know.
pub trait DeviceExt {
    /// The compute client of this device, initialized on first use.
    fn client(&self) -> Client;

    /// Whether a tensor with `shape` and `strides` can be read as it lies, or
    /// has to be made contiguous first. A property of the runtime, which is why
    /// it is asked of the device rather than of the client.
    fn can_read_tensor(
        &self,
        shape: &cubecl::zspace::Shape,
        strides: &cubecl::zspace::Strides,
    ) -> bool;
}

#[cfg(any_runtime)]
impl DeviceExt for Device {
    fn client(&self) -> Client {
        match *self {
            #[cfg(feature = "cuda")]
            Device::Cuda(ref device) => cubecl_cuda::CudaRuntime::client(device),
            #[cfg(feature = "hip")]
            Device::Hip(ref device) => cubecl_hip::HipRuntime::client(device),
            #[cfg(feature = "metal-native")]
            Device::Metal(ref device) => cubecl_metal::MetalRuntime::client(device),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(ref device) => <cubecl_wgpu::WgpuRuntime>::client(device),
            #[cfg(feature = "cpu")]
            Device::Cpu(ref device) => cubecl_cpu::CpuRuntime::client(device),
            #[allow(unreachable_patterns)]
            ref other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }

    fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool {
        match *self {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => cubecl_cuda::CudaRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "hip")]
            Device::Hip(_) => cubecl_hip::HipRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "metal-native")]
            Device::Metal(_) => cubecl_metal::MetalRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => <cubecl_wgpu::WgpuRuntime>::can_read_tensor(shape, strides),
            #[cfg(feature = "cpu")]
            Device::Cpu(_) => cubecl_cpu::CpuRuntime::can_read_tensor(shape, strides),
            #[allow(unreachable_patterns)]
            ref other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }
}

/// The default device of the most capable runtime this build links.
///
/// The order is the one a caller who did not choose would want — a discrete
/// accelerator over the portable path over the CPU — not the order the
/// features are declared in. `cubecl` itself has no `Default` for a
/// [`Device`]: which runtimes are linked is this crate's to know.
#[cfg(any_runtime)]
pub fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::Cuda(Default::default());
    #[cfg(all(feature = "hip", not(feature = "cuda")))]
    return Device::Hip(Default::default());
    #[cfg(all(feature = "metal-native", not(any(feature = "cuda", feature = "hip"))))]
    return Device::Metal(Default::default());
    #[cfg(all(
        feature = "wgpu",
        not(any(feature = "cuda", feature = "hip", feature = "metal-native"))
    ))]
    return Device::Wgpu(Default::default());
    #[cfg(all(
        feature = "cpu",
        not(any(
            feature = "cuda",
            feature = "hip",
            feature = "metal-native",
            feature = "wgpu"
        ))
    ))]
    return Device::Cpu(Default::default());
}

/// The ids of every device sharing `device_id`'s runtime and device type.
///
/// Takes and returns ids in [`Device`]'s own encoding, so a caller holding
/// one id can ask what else is on that device's runtime without unpacking
/// the runtime itself.
#[cfg(any_runtime)]
pub fn enumerate(device_id: DeviceId) -> alloc::vec::Vec<DeviceId> {
    let runtime = RuntimeId::of_device_id(device_id);
    let type_id = RuntimeId::strip(device_id).type_id;

    let ids = match runtime {
        #[cfg(feature = "cuda")]
        Ok(RuntimeId::Cuda) => cubecl_cuda::CudaRuntime::enumerate_devices(type_id),
        #[cfg(feature = "hip")]
        Ok(RuntimeId::Hip) => cubecl_hip::HipRuntime::enumerate_devices(type_id),
        #[cfg(feature = "metal-native")]
        Ok(RuntimeId::Metal) => cubecl_metal::MetalRuntime::enumerate_devices(type_id),
        #[cfg(feature = "wgpu")]
        Ok(RuntimeId::Wgpu) => <cubecl_wgpu::WgpuRuntime>::enumerate_devices(type_id),
        #[cfg(feature = "cpu")]
        Ok(RuntimeId::Cpu) => cubecl_cpu::CpuRuntime::enumerate_devices(type_id),
        _ => alloc::vec::Vec::new(),
    };

    match runtime {
        Ok(runtime) => ids.into_iter().map(|id| runtime.stamp(id)).collect(),
        Err(_) => ids,
    }
}

/// Every device of every runtime this build enables.
///
/// What a caller listing "the devices" wants, now that one type covers them
/// all — a runtime with no hardware present contributes nothing.
#[cfg(any_runtime)]
pub fn enumerate_all() -> alloc::vec::Vec<Device> {
    #[allow(unused_mut)]
    let mut devices = alloc::vec::Vec::new();

    #[cfg(feature = "cuda")]
    devices.extend(
        cubecl_cuda::CudaRuntime::enumerate_all_devices()
            .into_iter()
            .map(|id| Device::Cuda(DeviceIdentity::from_id(id))),
    );
    #[cfg(feature = "hip")]
    devices.extend(
        cubecl_hip::HipRuntime::enumerate_all_devices()
            .into_iter()
            .map(|id| Device::Hip(DeviceIdentity::from_id(id))),
    );
    #[cfg(feature = "metal-native")]
    devices.extend(
        cubecl_metal::MetalRuntime::enumerate_all_devices()
            .into_iter()
            .map(|id| Device::Metal(DeviceIdentity::from_id(id))),
    );
    #[cfg(feature = "wgpu")]
    devices.extend(
        <cubecl_wgpu::WgpuRuntime>::enumerate_all_devices()
            .into_iter()
            .map(|id| Device::Wgpu(DeviceIdentity::from_id(id))),
    );
    #[cfg(feature = "cpu")]
    devices.extend(
        cubecl_cpu::CpuRuntime::enumerate_all_devices()
            .into_iter()
            .map(|id| Device::Cpu(DeviceIdentity::from_id(id))),
    );

    devices
}

/// The runtime this build tests on.
#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_wgpu)]
pub type TestRuntime = wgpu::WgpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_cpu)]
pub type TestRuntime = cpu::CpuRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_cuda)]
pub type TestRuntime = cuda::CudaRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_hip)]
pub type TestRuntime = hip::HipRuntime;

/// The runtime this build tests on.
#[cfg(test_runtime_metal)]
pub type TestRuntime = metal::MetalRuntime;

/// The client of [`test_device`], for the test that only wants one.
///
/// ```no_run
/// let client = cubecl_dispatch::test_client();
/// ```
#[cfg(any(
    test_runtime_default,
    test_runtime_wgpu,
    test_runtime_cpu,
    test_runtime_cuda,
    test_runtime_hip,
    test_runtime_metal
))]
pub fn test_client() -> Client {
    test_device().client()
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
///
/// A test reaches its client through [`DeviceExt::client`] like any other
/// caller, which is what keeps it from naming a runtime to get one.
///
/// ```no_run
/// use cubecl_dispatch::DeviceExt;
///
/// let client = cubecl_dispatch::test_device().client();
/// ```
#[cfg(any(test_runtime_default, test_runtime_wgpu))]
pub fn test_device() -> Device {
    Device::Wgpu(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_cpu)]
pub fn test_device() -> Device {
    Device::Cpu(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_cuda)]
pub fn test_device() -> Device {
    Device::Cuda(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_hip)]
pub fn test_device() -> Device {
    Device::Hip(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_metal)]
pub fn test_device() -> Device {
    Device::Metal(Default::default())
}
