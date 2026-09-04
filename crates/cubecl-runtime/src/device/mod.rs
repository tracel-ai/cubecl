//! A device of any runtime, named without naming the runtime crate.
//!
//! Each runtime keeps its own device type — a value, nothing more — and this
//! module owns them all, so a caller can hold a [`Device`] without depending on
//! any runtime. Every variant is always there, whether or not the runtime is
//! in the build, which is what keeps this crate's shape the same in a library
//! build and in a binary that links a runtime.
//!
//! Reaching a client from a device, and choosing a default one, are the two
//! things this module cannot do on its own: both go through the runtimes, and
//! live in `cubecl-dispatch`.

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

use cubecl_common::device::Device as DeviceIdentity;
pub use cubecl_common::device::DeviceId;

/// A device of any runtime.
///
/// A value names both the runtime and the device on it. A value can name a
/// runtime the build does not link — the enum is the same in every build — and
/// it is reaching for a client that then fails, in `cubecl-dispatch`.
///
/// ```
/// use cubecl_runtime::device::{CudaDevice, Device};
///
/// let device = Device::Cuda(CudaDevice::new(0));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// A device of the CUDA runtime.
    Cuda(CudaDevice),
    /// A device of the HIP runtime.
    Hip(AmdDevice),
    /// A device of the native Metal runtime.
    Metal(MetalDevice),
    /// A device of the wgpu runtime, on its default compiler.
    Wgpu(WgpuDevice),
    /// The device of the CPU runtime.
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
            Self::Cuda(_) => RuntimeId::Cuda,
            Self::Hip(_) => RuntimeId::Hip,
            Self::Metal(_) => RuntimeId::Metal,
            Self::Wgpu(_) => RuntimeId::Wgpu,
            Self::Cpu(_) => RuntimeId::Cpu,
        }
    }
}

impl Device {
    /// The device a [`DeviceId`] in this type's encoding names.
    ///
    /// # Panics
    ///
    /// Where `device_id` names a runtime this build does not enable. An id only
    /// travels between builds of the same application, so this is a build
    /// mismatch rather than input to validate.
    pub fn from_id(device_id: DeviceId) -> Self {
        let runtime = RuntimeId::of_device_id(device_id);
        let inner = RuntimeId::strip(device_id);

        match runtime {
            Ok(RuntimeId::Cuda) => Self::Cuda(DeviceIdentity::from_id(inner)),
            Ok(RuntimeId::Hip) => Self::Hip(DeviceIdentity::from_id(inner)),
            Ok(RuntimeId::Metal) => Self::Metal(DeviceIdentity::from_id(inner)),
            Ok(RuntimeId::Wgpu) => Self::Wgpu(DeviceIdentity::from_id(inner)),
            Ok(RuntimeId::Cpu) => Self::Cpu(DeviceIdentity::from_id(inner)),
            Err(other) => {
                panic!("device id {device_id} names the runtime tag {other}, which no runtime has")
            }
        }
    }

    /// This device's id, with its runtime stamped in the high bits so ids from
    /// two runtimes cannot collide.
    pub fn to_id(&self) -> DeviceId {
        let inner = match *self {
            Self::Cuda(ref device) => DeviceIdentity::to_id(device),
            Self::Hip(ref device) => DeviceIdentity::to_id(device),
            Self::Metal(ref device) => DeviceIdentity::to_id(device),
            Self::Wgpu(ref device) => DeviceIdentity::to_id(device),
            Self::Cpu(ref device) => DeviceIdentity::to_id(device),
        };

        self.runtime().stamp(inner)
    }
}
impl From<CudaDevice> for Device {
    fn from(device: CudaDevice) -> Self {
        Self::Cuda(device)
    }
}
impl From<AmdDevice> for Device {
    fn from(device: AmdDevice) -> Self {
        Self::Hip(device)
    }
}
impl From<MetalDevice> for Device {
    fn from(device: MetalDevice) -> Self {
        Self::Metal(device)
    }
}
impl From<WgpuDevice> for Device {
    fn from(device: WgpuDevice) -> Self {
        Self::Wgpu(device)
    }
}
impl From<CpuDevice> for Device {
    fn from(device: CpuDevice) -> Self {
        Self::Cpu(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The id has to survive the trip in both directions, or a device stored by
    /// id comes back as a different device — on a different runtime, even.
    #[test]
    fn a_device_id_round_trips_through_its_runtime() {
        let device = Device::Wgpu(WgpuDevice::DiscreteGpu(1));

        let restored = Device::from_id(device.to_id());

        assert_eq!(device, restored);
    }

    /// The runtime rides in the high byte, which is what keeps two runtimes'
    /// devices from colliding on one id.
    #[test]
    fn a_device_id_names_its_runtime() {
        let device = Device::Wgpu(WgpuDevice::DiscreteGpu(1));

        let id = device.to_id();

        assert_eq!(RuntimeId::of_device_id(id), Ok(device.runtime()));
    }
}
