//! A device of any runtime, named without naming the runtime crate.
//!
//! Each runtime keeps its own device type — a value, nothing more — and this
//! module owns them all, so a caller can hold a [`Device`](self::Device) without depending on
//! any runtime. Which variants exist follows the `cuda`, `hip`, `metal`, `wgpu`
//! and `cpu` features, and each runtime crate turns its own on, so a variant is
//! there exactly when the runtime is in the build.
//!
//! Reaching a client from a device is the one thing this module cannot do on
//! its own: that goes through the runtime, and lives one crate up.

mod cpu;
mod cuda;
mod hip;
mod metal;
mod wgpu;

pub use cpu::CpuDevice;
pub use cuda::CudaDevice;
pub use hip::AmdDevice;
pub use metal::MetalDevice;
pub use wgpu::WgpuDevice;

#[cfg(any_runtime)]
use cubecl_common::device::Device as DeviceIdentity;
pub use cubecl_common::device::DeviceId;

/// A device of any runtime this build enables.
///
/// A value names both the runtime and the device on it. The variants follow
/// the runtime features, so a match on this type needs a wildcard arm.
///
/// ```no_run
/// # #[cfg(feature = "cuda")]
/// # fn main() {
/// use cubecl_runtime::device::{CudaDevice, Device};
///
/// let device = Device::Cuda(CudaDevice::new(0));
/// # }
/// # #[cfg(not(feature = "cuda"))]
/// # fn main() {}
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// A device of the CUDA runtime.
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
    /// A device of the HIP runtime.
    #[cfg(feature = "hip")]
    Hip(AmdDevice),
    /// A device of the native Metal runtime.
    #[cfg(feature = "metal")]
    Metal(MetalDevice),
    /// A device of the wgpu runtime, on its default compiler.
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuDevice),
    /// The device of the CPU runtime.
    #[cfg(feature = "cpu")]
    Cpu(CpuDevice),
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
const RUNTIME_TYPE_ID_MASK: u16 = 0x001F;
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

impl RuntimeId {
    /// The runtime a [`DeviceId`] in [`Device`]'s encoding names, or the raw
    /// tag when it names none this crate knows.
    pub fn of_device_id(device_id: DeviceId) -> Result<Self, u16> {
        Self::try_from(device_id.type_id >> RUNTIME_TYPE_ID_SHIFT)
    }

    /// Strip the runtime tag off a [`DeviceId`] in [`Device`]'s encoding,
    /// leaving the id the runtime itself hands out.
    pub fn strip(device_id: DeviceId) -> DeviceId {
        DeviceId::new(device_id.type_id & RUNTIME_TYPE_ID_MASK, device_id.index_id)
    }

    /// Stamp this runtime onto an id the runtime itself handed out.
    pub fn stamp(self, device_id: DeviceId) -> DeviceId {
        DeviceId::new(
            ((self as u16) << RUNTIME_TYPE_ID_SHIFT) | (device_id.type_id & RUNTIME_TYPE_ID_MASK),
            device_id.index_id,
        )
    }
}

impl Device {
    /// The runtime this device belongs to.
    pub fn runtime(&self) -> RuntimeId {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => RuntimeId::Cuda,
            #[cfg(feature = "hip")]
            Self::Hip(_) => RuntimeId::Hip,
            #[cfg(feature = "metal")]
            Self::Metal(_) => RuntimeId::Metal,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => RuntimeId::Wgpu,
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => RuntimeId::Cpu,
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
        #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "hip"))))]
        return Self::Metal(Default::default());
        #[cfg(all(
            feature = "wgpu",
            not(any(feature = "cuda", feature = "hip", feature = "metal"))
        ))]
        return Self::Wgpu(Default::default());
        #[cfg(all(
            feature = "cpu",
            not(any(feature = "cuda", feature = "hip", feature = "metal", feature = "wgpu"))
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
        let runtime = RuntimeId::of_device_id(device_id);
        let inner = RuntimeId::strip(device_id);

        match runtime {
            #[cfg(feature = "cuda")]
            Ok(RuntimeId::Cuda) => Self::Cuda(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "hip")]
            Ok(RuntimeId::Hip) => Self::Hip(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "metal")]
            Ok(RuntimeId::Metal) => Self::Metal(DeviceIdentity::from_id(inner)),
            #[cfg(feature = "wgpu")]
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
            #[cfg(feature = "metal")]
            Self::Metal(ref device) => DeviceIdentity::to_id(device),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(ref device) => DeviceIdentity::to_id(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(ref device) => DeviceIdentity::to_id(device),
        };

        self.runtime().stamp(inner)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaDevice> for Device {
    fn from(device: CudaDevice) -> Self {
        Self::Cuda(device)
    }
}

#[cfg(feature = "hip")]
impl From<AmdDevice> for Device {
    fn from(device: AmdDevice) -> Self {
        Self::Hip(device)
    }
}

#[cfg(feature = "metal")]
impl From<MetalDevice> for Device {
    fn from(device: MetalDevice) -> Self {
        Self::Metal(device)
    }
}

#[cfg(feature = "wgpu")]
impl From<WgpuDevice> for Device {
    fn from(device: WgpuDevice) -> Self {
        Self::Wgpu(device)
    }
}

#[cfg(feature = "cpu")]
impl From<CpuDevice> for Device {
    fn from(device: CpuDevice) -> Self {
        Self::Cpu(device)
    }
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

        assert_eq!(RuntimeId::of_device_id(id), Ok(device.runtime()));
    }
}
