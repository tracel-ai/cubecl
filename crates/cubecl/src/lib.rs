extern crate alloc;

pub use cubecl_core::*;

use cubecl_core::client::Client;
#[cfg(any_runtime)]
use cubecl_core::device::{Device as DeviceIdentity, DeviceId};
#[cfg(any_runtime)]
use cubecl_runtime::runtime::Runtime;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

/// Ship pre-warmed autotune and compilation caches with an application.
///
/// Run the application once so its caches are warm, then save the active
/// environment to a file and ship it. On the target machine either
/// [`environment::load`] mounts the file in place, or [`import`] copies it
/// into the local environment, after which the file can be deleted.
///
/// The exporting binary must be built against the same cubecl version as the
/// consuming one. The version is part of every cache namespace, so a bundle
/// built elsewhere installs cleanly and is then ignored, with a warning as the
/// only signal. Calling this from your own crate is what keeps the two in
/// step.
///
/// ```no_run
/// use cubecl::bundle::BundleFormat;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // ... run the work you want tuned and compiled, then:
/// let manifest = cubecl::environment::bundle().save("h100.bundle", BundleFormat::Sqlite)?;
/// println!("exported {}", manifest.name);
/// # Ok(())
/// # }
/// ```
///
/// Merging several roots or restricting the namespaces goes through
/// [`export`] directly.
///
/// [`export`]: cubecl_environment::bundle::export
/// [`import`]: cubecl_environment::bundle::import
/// [`environment::load`]: cubecl_environment::environment::load
pub use cubecl_environment::bundle;

/// Which named environment caches are warmed into, and where it lives.
pub use cubecl_environment::environment;

/// Running a workload for the compilation and tuning it provokes, without
/// running the workload itself.
///
/// This is what makes producing a [`bundle`] affordable: inside a
/// [`DryRun`](dry_run::DryRun) every launch is compiled, cached and tuned
/// without also being executed. Buffers are left as they were, so it only suits
/// a pass driven by the *shapes* it produces.
///
/// ```no_run
/// # fn warm_up() {}
/// let _dry_run = cubecl::dry_run::DryRun::new();
/// warm_up();
/// ```
pub use cubecl_runtime::dry_run;

/// Watching what the runtime runs: the launch observer, and the profiling
/// logger's levels.
///
/// Re-exported because a caller attributing kernels to its own work reaches for
/// [`LaunchObservation`](cubecl_runtime::logging::LaunchObservation) and has no
/// other reason to depend on `cubecl-runtime` directly.
pub use cubecl_runtime::logging;

/// A device of any runtime this build enables.
///
/// Each runtime has its own device type, and until now the only way to reach a
/// [`Client`] was through the runtime, `R::client(&device)`. This enum
/// is the runtime-independent way: a value names both the runtime and the
/// device on it, and [`client`](Device::client) hands back the client for it.
///
/// The variants follow the runtime features, so a match on this type needs a
/// wildcard arm.
///
/// ```no_run
/// # #[cfg(feature = "cuda")]
/// # fn main() {
/// use cubecl::Device;
///
/// let device = Device::Cuda(cubecl::cuda::CudaDevice::new(0));
/// let client = device.client();
/// println!("running on {}", client.name());
/// # }
/// # #[cfg(not(feature = "cuda"))]
/// # fn main() {}
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// A device of the CUDA runtime.
    #[cfg(feature = "cuda")]
    Cuda(cubecl_cuda::CudaDevice),
    /// A device of the HIP runtime.
    #[cfg(feature = "hip")]
    Hip(cubecl_hip::AmdDevice),
    /// A device of the native Metal runtime.
    #[cfg(feature = "metal-native")]
    Metal(cubecl_metal::MetalDevice),
    /// A device of the wgpu runtime, on its default compiler.
    #[cfg(any(feature = "wgpu", test_runtime_default))]
    Wgpu(cubecl_wgpu::WgpuDevice),
    /// The device of the CPU runtime.
    #[cfg(feature = "cpu")]
    Cpu(cubecl_cpu::CpuDevice),
}

/// Which runtime a [`Device`] belongs to.
///
/// The discriminants are fixed per runtime rather than assigned by whichever
/// runtimes a build enables, so a [`DeviceId`] means the same thing in every
/// build that can read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum RuntimeId {
    /// The CUDA runtime.
    Cuda = 0,
    /// The HIP runtime.
    Hip = 1,
    /// The native Metal runtime.
    Metal = 2,
    /// The wgpu runtime.
    Wgpu = 3,
    /// The CPU runtime.
    Cpu = 4,
}

/// A runtime's own device types get the low five bits of a [`DeviceId`]'s type
/// id, the runtime itself the three above them. That split is what lets one id
/// name a device across every runtime, and it stays inside one byte on purpose:
/// a caller that nests these ids in an encoding of its own — burn's dispatch
/// backend does — still has the high byte to spend.
///
/// Five bits is room for 32 device types per runtime, against the six the
/// widest runtime here declares.
#[cfg(any_runtime)]
const RUNTIME_TYPE_ID_MASK: u16 = 0x001F;
#[cfg(any_runtime)]
const RUNTIME_TYPE_ID_SHIFT: u32 = 5;

impl TryFrom<u16> for RuntimeId {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Cuda),
            1 => Ok(Self::Hip),
            2 => Ok(Self::Metal),
            3 => Ok(Self::Wgpu),
            4 => Ok(Self::Cpu),
            other => Err(other),
        }
    }
}

impl Device {
    /// Whether a tensor with `shape` and `strides` can be read as it lies, or
    /// has to be made contiguous first. A property of the runtime, which is why
    /// it is asked of the device rather than of the client.
    #[cfg(any_runtime)]
    pub fn can_read_tensor(
        &self,
        shape: &cubecl_core::zspace::Shape,
        strides: &cubecl_core::zspace::Strides,
    ) -> bool {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => cubecl_cuda::CudaRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "hip")]
            Self::Hip(_) => cubecl_hip::HipRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "metal-native")]
            Self::Metal(_) => cubecl_metal::MetalRuntime::can_read_tensor(shape, strides),
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Self::Wgpu(_) => <cubecl_wgpu::WgpuRuntime>::can_read_tensor(shape, strides),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => cubecl_cpu::CpuRuntime::can_read_tensor(shape, strides),
        }
    }

    /// The ids of every device sharing `device_id`'s runtime and device type.
    ///
    /// Takes and returns ids in [`Device`]'s own encoding, so a caller holding
    /// one id can ask what else is on that device's runtime without unpacking
    /// the runtime itself.
    #[cfg(any_runtime)]
    pub fn enumerate(device_id: DeviceId) -> alloc::vec::Vec<DeviceId> {
        let runtime = RuntimeId::try_from(device_id.type_id >> RUNTIME_TYPE_ID_SHIFT);
        let type_id = device_id.type_id & RUNTIME_TYPE_ID_MASK;

        let ids = match runtime {
            #[cfg(feature = "cuda")]
            Ok(RuntimeId::Cuda) => cubecl_cuda::CudaRuntime::enumerate_devices(type_id),
            #[cfg(feature = "hip")]
            Ok(RuntimeId::Hip) => cubecl_hip::HipRuntime::enumerate_devices(type_id),
            #[cfg(feature = "metal-native")]
            Ok(RuntimeId::Metal) => cubecl_metal::MetalRuntime::enumerate_devices(type_id),
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Ok(RuntimeId::Wgpu) => <cubecl_wgpu::WgpuRuntime>::enumerate_devices(type_id),
            #[cfg(feature = "cpu")]
            Ok(RuntimeId::Cpu) => cubecl_cpu::CpuRuntime::enumerate_devices(type_id),
            _ => alloc::vec::Vec::new(),
        };

        let tag = device_id.type_id & !RUNTIME_TYPE_ID_MASK;
        ids.into_iter()
            .map(|id| DeviceId::new(tag | (id.type_id & RUNTIME_TYPE_ID_MASK), id.index_id))
            .collect()
    }

    /// Every device of every runtime this build enables.
    ///
    /// What a caller listing "the devices" wants, now that one type covers them
    /// all — a runtime with no hardware present contributes nothing.
    #[cfg(any_runtime)]
    pub fn enumerate_all() -> alloc::vec::Vec<Self> {
        #[allow(unused_mut)]
        let mut devices = alloc::vec::Vec::new();

        #[cfg(feature = "cuda")]
        devices.extend(
            cubecl_cuda::CudaRuntime::enumerate_all_devices()
                .into_iter()
                .map(|id| Self::Cuda(DeviceIdentity::from_id(id))),
        );
        #[cfg(feature = "hip")]
        devices.extend(
            cubecl_hip::HipRuntime::enumerate_all_devices()
                .into_iter()
                .map(|id| Self::Hip(DeviceIdentity::from_id(id))),
        );
        #[cfg(feature = "metal-native")]
        devices.extend(
            cubecl_metal::MetalRuntime::enumerate_all_devices()
                .into_iter()
                .map(|id| Self::Metal(DeviceIdentity::from_id(id))),
        );
        #[cfg(any(feature = "wgpu", test_runtime_default))]
        devices.extend(
            <cubecl_wgpu::WgpuRuntime>::enumerate_all_devices()
                .into_iter()
                .map(|id| Self::Wgpu(DeviceIdentity::from_id(id))),
        );
        #[cfg(feature = "cpu")]
        devices.extend(
            cubecl_cpu::CpuRuntime::enumerate_all_devices()
                .into_iter()
                .map(|id| Self::Cpu(DeviceIdentity::from_id(id))),
        );

        devices
    }

    /// The runtime this device belongs to.
    pub fn runtime(&self) -> RuntimeId {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => RuntimeId::Cuda,
            #[cfg(feature = "hip")]
            Self::Hip(_) => RuntimeId::Hip,
            #[cfg(feature = "metal-native")]
            Self::Metal(_) => RuntimeId::Metal,
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Self::Wgpu(_) => RuntimeId::Wgpu,
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => RuntimeId::Cpu,
        }
    }
}

impl Device {
    /// The compute client of this device, initialized on first use.
    pub fn client(&self) -> Client {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ref device) => cubecl_cuda::CudaRuntime::client(device),
            #[cfg(feature = "hip")]
            Self::Hip(ref device) => cubecl_hip::HipRuntime::client(device),
            #[cfg(feature = "metal-native")]
            Self::Metal(ref device) => cubecl_metal::MetalRuntime::client(device),
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Self::Wgpu(ref device) => <cubecl_wgpu::WgpuRuntime>::client(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(ref device) => cubecl_cpu::CpuRuntime::client(device),
        }
    }
}

#[cfg(any_runtime)]
impl Default for Device {
    /// The default device of the most capable runtime this build enables.
    ///
    /// The order is the one a caller who did not choose would want — a discrete
    /// accelerator over the portable path over the CPU — not the order the
    /// features are declared in.
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        return Self::Cuda(Default::default());
        #[cfg(all(feature = "hip", not(feature = "cuda")))]
        return Self::Hip(Default::default());
        #[cfg(all(feature = "metal-native", not(any(feature = "cuda", feature = "hip"))))]
        return Self::Metal(Default::default());
        #[cfg(all(
            any(feature = "wgpu", test_runtime_default),
            not(any(feature = "cuda", feature = "hip", feature = "metal-native"))
        ))]
        return Self::Wgpu(Default::default());
        #[cfg(all(
            feature = "cpu",
            not(any(
                feature = "cuda",
                feature = "hip",
                feature = "metal-native",
                feature = "wgpu",
                test_runtime_default
            ))
        ))]
        return Self::Cpu(Default::default());
    }
}

#[cfg(any_runtime)]
impl DeviceIdentity for Device {
    /// # Panics
    ///
    /// Where `device_id` names a runtime this build does not enable. An id only
    /// travels between builds of the same application, so this is a build
    /// mismatch rather than input to validate.
    fn from_id(device_id: DeviceId) -> Self {
        let runtime = RuntimeId::try_from(device_id.type_id >> RUNTIME_TYPE_ID_SHIFT);
        let inner = DeviceId::new(device_id.type_id & RUNTIME_TYPE_ID_MASK, device_id.index_id);

        match runtime {
            #[cfg(feature = "cuda")]
            Ok(RuntimeId::Cuda) => Self::Cuda(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "hip")]
            Ok(RuntimeId::Hip) => Self::Hip(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "metal-native")]
            Ok(RuntimeId::Metal) => Self::Metal(DeviceIdentity::from_id(inner)),
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Ok(RuntimeId::Wgpu) => Self::Wgpu(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "cpu")]
            Ok(RuntimeId::Cpu) => Self::Cpu(DeviceIdentity::from_id(inner)),
            other => panic!(
                "device id {device_id} names the runtime {other:?}, which this build does not enable"
            ),
        }
    }

    fn to_id(&self) -> DeviceId {
        let inner = match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ref device) => DeviceIdentity::to_id(device),
            #[cfg(feature = "hip")]
            Self::Hip(ref device) => DeviceIdentity::to_id(device),
            #[cfg(feature = "metal-native")]
            Self::Metal(ref device) => DeviceIdentity::to_id(device),
            #[cfg(any(feature = "wgpu", test_runtime_default))]
            Self::Wgpu(ref device) => DeviceIdentity::to_id(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(ref device) => DeviceIdentity::to_id(device),
        };

        DeviceId::new(
            ((self.runtime() as u16) << RUNTIME_TYPE_ID_SHIFT)
                | (inner.type_id & RUNTIME_TYPE_ID_MASK),
            inner.index_id,
        )
    }
}

#[cfg(feature = "cuda")]
impl From<cubecl_cuda::CudaDevice> for Device {
    fn from(device: cubecl_cuda::CudaDevice) -> Self {
        Self::Cuda(device)
    }
}

#[cfg(feature = "hip")]
impl From<cubecl_hip::AmdDevice> for Device {
    fn from(device: cubecl_hip::AmdDevice) -> Self {
        Self::Hip(device)
    }
}

#[cfg(feature = "metal-native")]
impl From<cubecl_metal::MetalDevice> for Device {
    fn from(device: cubecl_metal::MetalDevice) -> Self {
        Self::Metal(device)
    }
}

#[cfg(any(feature = "wgpu", test_runtime_default))]
impl From<cubecl_wgpu::WgpuDevice> for Device {
    fn from(device: cubecl_wgpu::WgpuDevice) -> Self {
        Self::Wgpu(device)
    }
}

#[cfg(feature = "cpu")]
impl From<cubecl_cpu::CpuDevice> for Device {
    fn from(device: cubecl_cpu::CpuDevice) -> Self {
        Self::Cpu(device)
    }
}

#[cfg(feature = "wgpu")]
pub use cubecl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "hip")]
pub use cubecl_hip as hip;

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;

#[cfg(feature = "cpu")]
pub use cubecl_cpu as cpu;

#[cfg(feature = "metal-native")]
pub use cubecl_metal as metal;

#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(test_runtime_wgpu)]
pub type TestRuntime = wgpu::WgpuRuntime;

#[cfg(test_runtime_cpu)]
pub type TestRuntime = cpu::CpuRuntime;

#[cfg(test_runtime_cuda)]
pub type TestRuntime = cuda::CudaRuntime;

#[cfg(test_runtime_hip)]
pub type TestRuntime = hip::HipRuntime;

#[cfg(test_runtime_metal)]
pub type TestRuntime = metal::MetalRuntime;

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
///
/// A test reaches its client through [`Device::client`] like any other caller,
/// which is what keeps it from naming a runtime — or the [`Runtime`] trait —
/// to get one.
///
/// ```no_run
/// let client = cubecl::test_device().client();
/// ```
#[cfg(test_runtime_default)]
pub fn test_device() -> Device {
    Device::Wgpu(Default::default())
}

/// The [`Device`] of [`TestRuntime`], the runtime this build tests on.
#[cfg(test_runtime_wgpu)]
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

#[cfg(all(test, any_runtime))]
mod tests {
    use super::*;

    /// The id has to survive the trip in both directions, or a device stored by
    /// id comes back as a different device — on a different runtime, even.
    #[test]
    fn a_device_id_round_trips_through_its_runtime() {
        let device = Device::default();

        let restored = <Device as DeviceIdentity>::from_id(DeviceIdentity::to_id(&device));

        assert_eq!(device, restored);
    }

    /// The runtime rides in the high byte, which is what keeps two runtimes'
    /// devices from colliding on one id.
    #[test]
    fn a_device_id_names_its_runtime() {
        let device = Device::default();

        let id = DeviceIdentity::to_id(&device);

        assert_eq!(
            RuntimeId::try_from(id.type_id >> RUNTIME_TYPE_ID_SHIFT),
            Ok(device.runtime())
        );
    }
}
