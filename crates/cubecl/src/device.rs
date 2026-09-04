//! A device of any runtime, and the runtime it belongs to.
//!
//! Each runtime keeps its own device type — a value, nothing more, and
//! `cubecl-runtime` owns them all. This module puts one enum over them, so a
//! caller names a device without naming a runtime, and turns that device into
//! a [`Client`](crate::client::Client). Every variant is there whether or not the
//! build links the runtime, which is what keeps a stored id meaning the same
//! thing everywhere; reaching for a client of a runtime this build left out is
//! what fails.

use cubecl_core::client::Client;
#[cfg(any_runtime)]
use cubecl_core::device::Device as DeviceIdentity;
use cubecl_core::zspace::{Shape, Strides};
#[cfg(any_runtime)]
use cubecl_runtime::runtime::Runtime;

pub use cubecl_core::device::DeviceId;
pub use cubecl_runtime::device::{AmdDevice, CpuDevice, CudaDevice, MetalDevice, WgpuDevice};

/// A device of any runtime.
///
/// A value names both the runtime and the device on it. A value can name a
/// runtime the build does not link — the enum is the same in every build — and
/// it is reaching for a client that then fails.
///
/// ```no_run
/// # #[cfg(feature = "wgpu")]
/// # fn main() {
/// use cubecl::Device;
///
/// let client = Device::default().client();
/// println!("running on {}", client.name());
/// # }
/// # #[cfg(not(feature = "wgpu"))]
/// # fn main() {}
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
/// The runtime tag, once shifted down: three bits, room for eight runtimes.
const RUNTIME_MASK: u16 = 0x0007;

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
    /// The bits of a [`DeviceId`]'s type id neither a runtime nor its device
    /// types claim: the high byte, which is a caller nesting these ids in an
    /// encoding of its own to spend. Read them off an id before handing it to
    /// a runtime, and put them back on what comes out.
    pub const OUTER_MASK: u16 = !(RUNTIME_TYPE_ID_MASK | (RUNTIME_MASK << RUNTIME_TYPE_ID_SHIFT));

    /// The runtime a [`DeviceId`] in [`Device`]'s encoding names, or the raw
    /// tag when it names none this crate knows.
    ///
    /// Reads the three tag bits alone, so an id carrying a caller's own bits
    /// in the high byte names the same runtime as one without them.
    pub fn of_device_id(device_id: DeviceId) -> Result<Self, u16> {
        Self::try_from((device_id.type_id >> RUNTIME_TYPE_ID_SHIFT) & RUNTIME_MASK)
    }

    /// Strip the runtime tag off a [`DeviceId`] in [`Device`]'s encoding,
    /// leaving the id the runtime itself hands out — the low five bits, and
    /// nothing of the high byte a caller may have spent on its own encoding.
    pub fn strip(device_id: DeviceId) -> DeviceId {
        DeviceId::new(device_id.type_id & RUNTIME_TYPE_ID_MASK, device_id.index_id)
    }

    /// Stamp this runtime onto an id the runtime itself handed out.
    ///
    /// Only the three tag bits are written, so whatever `device_id` holds in
    /// the high byte survives the round trip.
    pub fn stamp(self, device_id: DeviceId) -> DeviceId {
        let tag = ((self as u16) & RUNTIME_MASK) << RUNTIME_TYPE_ID_SHIFT;
        DeviceId::new(
            (device_id.type_id & !(RUNTIME_MASK << RUNTIME_TYPE_ID_SHIFT)) | tag,
            device_id.index_id,
        )
    }
}

impl Device {
    /// The compute client of this device, initialized on first use.
    ///
    /// # Panics
    ///
    /// Where this build does not link the device's runtime.
    pub fn client(&self) -> Client {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ref device) => cubecl_cuda::CudaRuntime::client(device),
            #[cfg(feature = "hip")]
            Self::Hip(ref device) => cubecl_hip::HipRuntime::client(device),
            #[cfg(feature = "metal-native")]
            Self::Metal(ref device) => cubecl_metal::MetalRuntime::client(device),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(ref device) => <cubecl_wgpu::WgpuRuntime>::client(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(ref device) => cubecl_cpu::CpuRuntime::client(device),
            #[allow(unreachable_patterns)]
            ref other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }

    /// Whether a tensor with `shape` and `strides` can be read as it lies, or
    /// has to be made contiguous first. A property of the runtime, which is why
    /// it is asked of the device rather than of the client.
    ///
    /// # Panics
    ///
    /// Where this build does not link the device's runtime.
    #[cfg_attr(not(any_runtime), allow(unused_variables))]
    pub fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => cubecl_cuda::CudaRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "hip")]
            Self::Hip(_) => cubecl_hip::HipRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "metal-native")]
            Self::Metal(_) => cubecl_metal::MetalRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => <cubecl_wgpu::WgpuRuntime>::can_read_tensor(shape, strides),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => cubecl_cpu::CpuRuntime::can_read_tensor(shape, strides),
            #[allow(unreachable_patterns)]
            ref other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }

    /// The ids of every device sharing `device_id`'s runtime and device type.
    ///
    /// Takes and returns ids in [`Device`]'s own encoding, so a caller holding
    /// one id can ask what else is on that device's runtime without unpacking
    /// the runtime itself. Whatever `device_id` spends the high byte on comes
    /// back on every id, untouched.
    pub fn enumerate(device_id: DeviceId) -> alloc::vec::Vec<DeviceId> {
        let runtime = RuntimeId::of_device_id(device_id);
        #[cfg_attr(not(any_runtime), allow(unused_variables))]
        let type_id = RuntimeId::strip(device_id).type_id;
        let outer = device_id.type_id & RuntimeId::OUTER_MASK;

        let ids: alloc::vec::Vec<DeviceId> = match runtime {
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
            Ok(runtime) => ids
                .into_iter()
                .map(|id| runtime.stamp(DeviceId::new(id.type_id | outer, id.index_id)))
                .collect(),
            Err(_) => ids,
        }
    }

    /// Every device of every runtime this build links.
    ///
    /// What a caller listing "the devices" wants, now that one type covers them
    /// all — a runtime with no hardware present contributes nothing.
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
        #[cfg(feature = "wgpu")]
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
            Self::Cuda(_) => RuntimeId::Cuda,
            Self::Hip(_) => RuntimeId::Hip,
            Self::Metal(_) => RuntimeId::Metal,
            Self::Wgpu(_) => RuntimeId::Wgpu,
            Self::Cpu(_) => RuntimeId::Cpu,
        }
    }

    /// The device a [`DeviceId`] in this type's encoding names.
    ///
    /// # Panics
    ///
    /// Where `device_id` names no runtime at all. An id only travels between
    /// builds of the same application, so this is a build mismatch rather than
    /// input to validate.
    pub fn from_id(device_id: DeviceId) -> Self {
        let runtime = RuntimeId::of_device_id(device_id);
        let inner = RuntimeId::strip(device_id);

        match runtime {
            Ok(RuntimeId::Cuda) => Self::Cuda(cubecl_core::device::Device::from_id(inner)),
            Ok(RuntimeId::Hip) => Self::Hip(cubecl_core::device::Device::from_id(inner)),
            Ok(RuntimeId::Metal) => Self::Metal(cubecl_core::device::Device::from_id(inner)),
            Ok(RuntimeId::Wgpu) => Self::Wgpu(cubecl_core::device::Device::from_id(inner)),
            Ok(RuntimeId::Cpu) => Self::Cpu(cubecl_core::device::Device::from_id(inner)),
            Err(other) => {
                panic!("device id {device_id} names the runtime tag {other}, which no runtime has")
            }
        }
    }

    /// This device's id, with its runtime stamped in the tag bits so ids from
    /// two runtimes cannot collide.
    pub fn to_id(&self) -> DeviceId {
        use cubecl_core::device::Device as _;

        let inner = match *self {
            Self::Cuda(ref device) => device.to_id(),
            Self::Hip(ref device) => device.to_id(),
            Self::Metal(ref device) => device.to_id(),
            Self::Wgpu(ref device) => device.to_id(),
            Self::Cpu(ref device) => device.to_id(),
        };

        self.runtime().stamp(inner)
    }
}

/// The runtimes this build links, most capable first.
///
/// A caller who did not choose wants a discrete accelerator over the portable
/// path over the CPU, which is not the order the features are declared in.
#[cfg(any_runtime)]
const LINKED: &[RuntimeId] = &[
    #[cfg(feature = "cuda")]
    RuntimeId::Cuda,
    #[cfg(feature = "hip")]
    RuntimeId::Hip,
    #[cfg(feature = "metal-native")]
    RuntimeId::Metal,
    #[cfg(feature = "wgpu")]
    RuntimeId::Wgpu,
    #[cfg(feature = "cpu")]
    RuntimeId::Cpu,
];

#[cfg(any_runtime)]
impl RuntimeId {
    /// Whether this machine has hardware worth choosing this runtime for.
    ///
    /// Each runtime answers for itself — zero devices rather than a failure
    /// when its driver is missing, and no claim on a machine where all it has
    /// is a software fallback.
    fn is_available(self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => cubecl_cuda::CudaRuntime::is_available(),
            #[cfg(feature = "hip")]
            Self::Hip => cubecl_hip::HipRuntime::is_available(),
            #[cfg(feature = "metal-native")]
            Self::Metal => cubecl_metal::MetalRuntime::is_available(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu => <cubecl_wgpu::WgpuRuntime>::is_available(),
            #[cfg(feature = "cpu")]
            Self::Cpu => cubecl_cpu::CpuRuntime::is_available(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// This runtime's own default device — which of its devices is best is the
    /// runtime's business, not this crate's.
    fn default_device(self) -> Device {
        match self {
            Self::Cuda => Device::Cuda(Default::default()),
            Self::Hip => Device::Hip(Default::default()),
            Self::Metal => Device::Metal(Default::default()),
            Self::Wgpu => Device::Wgpu(Default::default()),
            Self::Cpu => Device::Cpu(Default::default()),
        }
    }
}

/// No answer yet. No real id can collide with it: [`Device::to_id`] writes a
/// runtime tag of 4 or less into bits 5 to 7, and this has all three set.
#[cfg(any_runtime)]
const UNPROBED: u32 = u32::MAX;

/// What [`Device::default`] settled on, packed as its [`DeviceId`].
///
/// Two threads racing here both run the same walk and store the same answer,
/// so the only cost of losing the race is having done the work twice.
#[cfg(any_runtime)]
static DEFAULT_DEVICE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(UNPROBED);

/// The default device of the most capable runtime with hardware to run on.
///
/// Walks the runtimes this build links, most capable first, and takes the
/// first that reports a device — so a binary that links both CUDA and wgpu
/// lands on wgpu when it turns out there is no NVIDIA card, rather than
/// naming a device it cannot reach.
///
/// The last runtime in that order is taken without asking. There is nothing
/// left to fall back to, so its answer cannot change the outcome, and a build
/// that links one runtime — the usual one — probes nothing at all. Where a
/// probe does run, its result is kept for the life of the process: a device
/// appearing or disappearing later goes unnoticed.
#[cfg(any_runtime)]
impl Default for Device {
    fn default() -> Self {
        use core::sync::atomic::Ordering;

        let cached = DEFAULT_DEVICE.load(Ordering::Relaxed);
        if cached != UNPROBED {
            return Self::from_id(DeviceId::new((cached >> 16) as u16, cached as u16));
        }

        let device = Self::probe_default();
        let id = device.to_id();
        DEFAULT_DEVICE.store(
            ((id.type_id as u32) << 16) | id.index_id as u32,
            Ordering::Relaxed,
        );

        device
    }
}

#[cfg(any_runtime)]
impl Device {
    /// The walk [`Device::default`] caches the answer to.
    fn probe_default() -> Self {
        let (last, rest) = LINKED
            .split_last()
            .expect("`any_runtime` is set, so this build links at least one runtime");

        for runtime in rest {
            if runtime.is_available() {
                return runtime.default_device();
            }
        }

        last.default_device()
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

    /// The runtime rides in the three bits above the device type, which is
    /// what keeps two runtimes' devices from colliding on one id.
    #[test]
    fn a_device_id_names_its_runtime() {
        let device = Device::Wgpu(WgpuDevice::DiscreteGpu(1));

        let id = device.to_id();

        assert_eq!(RuntimeId::of_device_id(id), Ok(device.runtime()));
    }

    /// The high byte is a nesting caller's to spend, so an id carrying one
    /// still names its runtime and still restores its device.
    #[test]
    fn a_nested_high_byte_does_not_change_what_an_id_names() {
        let device = Device::Wgpu(WgpuDevice::DiscreteGpu(1));
        let id = device.to_id();

        let nested = DeviceId::new(id.type_id | 0xAB00, id.index_id);

        assert_eq!(RuntimeId::of_device_id(nested), Ok(device.runtime()));
        assert_eq!(RuntimeId::strip(nested), RuntimeId::strip(id));
        assert_eq!(Device::from_id(nested), device);
    }

    /// And stamping a runtime on writes the tag bits alone, so the high byte
    /// comes back out of a round trip untouched.
    #[test]
    fn stamping_a_runtime_leaves_the_high_byte_alone() {
        let inner = DeviceId::new(0x03, 7);

        let stamped = RuntimeId::Cpu.stamp(DeviceId::new(inner.type_id | 0xAB00, inner.index_id));

        assert_eq!(stamped.type_id & RuntimeId::OUTER_MASK, 0xAB00);
        assert_eq!(RuntimeId::of_device_id(stamped), Ok(RuntimeId::Cpu));
        assert_eq!(RuntimeId::strip(stamped), inner);
    }
}

#[cfg(all(test, any_runtime))]
mod default_tests {
    use super::*;

    /// Where this machine has hardware for any linked runtime, the walk has to
    /// land on one of them, or the client call right after it fails on a device
    /// nothing can reach. A build whose runtimes are all absent has no better
    /// answer to give, and still has to give one.
    #[test]
    fn the_default_device_is_one_this_machine_has() {
        let device = Device::default();

        if LINKED.iter().any(|runtime| runtime.is_available()) {
            assert!(
                device.runtime().is_available(),
                "{device:?} was chosen over a runtime this machine can actually run"
            );
        }
    }

    /// The second call reads the cache rather than walking again, so it has to
    /// decode back to the same device.
    #[test]
    fn the_cached_default_decodes_back_to_the_same_device() {
        let first = Device::default();
        let second = Device::default();

        assert_eq!(first, second);
        assert_eq!(first.to_id(), second.to_id());
    }

    /// The sentinel has to be a value no real device can encode to, or the
    /// first device to hit it would be re-probed on every call.
    #[test]
    fn no_device_encodes_to_the_unprobed_sentinel() {
        for runtime in LINKED {
            let id = runtime.default_device().to_id();

            let encoded = ((id.type_id as u32) << 16) | id.index_id as u32;

            assert_ne!(encoded, UNPROBED);
        }
    }
}
