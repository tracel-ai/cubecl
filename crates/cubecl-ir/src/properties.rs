use alloc::string::String;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::str::FromStr;

use crate::EnumSet;
use crate::EnumSetType;
use crate::{
    AddressType, ElemType, OpaqueType, SemanticType, Type, TypeHash, VectorSize,
    features::{AtomicUsage, ComplexUsage, Features, TypeUsage},
};
use cubecl_common::profile::TimingMethod;

/// Properties of the device related to the accelerator hardware.
///
/// # Plane size min/max
///
/// This is a range of possible values for the plane size.
///
/// For Nvidia GPUs and HIP, this is a single fixed value.
///
/// For wgpu with AMD GPUs this is a range of possible values, but the actual configured value
/// is undefined and can only be queried at runtime. Should usually be 32, but not guaranteed.
///
/// For Intel GPUs, this is variable based on the number of registers used in the kernel. No way to
/// query this at compile time is currently available. As a result, the minimum value should usually
/// be assumed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HardwareProperties {
    /// The maximum size of a single load instruction, in bits. Used for optimized vector sizes.
    pub load_width: u32,
    /// The minimum size of a plane on this device
    pub plane_size_min: u32,
    /// The maximum size of a plane on this device
    pub plane_size_max: u32,
    /// minimum number of bindings for a kernel that can be used at once.
    pub max_bindings: u32,
    /// Maximum amount of shared memory, in bytes
    pub max_shared_memory_size: usize,
    /// Maximum `CubeCount` in x, y and z dimensions
    pub max_cube_count: (u32, u32, u32),
    /// Maximum number of total units in a cube
    pub max_units_per_cube: u32,
    /// Maximum `CubeDim` in x, y, and z dimensions
    pub max_cube_dim: (u32, u32, u32),
    /// Number of streaming multiprocessors (SM), if available
    pub num_streaming_multiprocessors: Option<u32>,
    /// Number of available parallel cpu units, if the runtime is CPU.
    pub num_cpu_cores: Option<u32>,
    /// Bytes of the device's last level cache, and `None`, never `Some(0)`,
    /// where the runtime cannot read one.
    ///
    /// The size a working set has to outgrow before what it reaches is set by
    /// memory rather than by the chip.
    pub last_level_cache_size: Option<usize>,
    /// Number of tensor cores per SM, if any
    pub num_tensor_cores: Option<u32>,
    /// The minimum tiling dimension for a single axis in tensor cores.
    ///
    /// For a backend that only supports 16x16x16, the value would be 16.
    /// For a backend that also supports 32x8x16, the value would be 8.
    pub min_tensor_cores_dim: Option<u32>,
    /// Maximum vector size supported by the device
    pub max_vector_size: VectorSize,
    /// Memory reserved for the driver when using cube-scoped matrices
    pub cube_mma_reserved_shared_memory: usize,
}

/// Properties of the device related to allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryDeviceProperties {
    /// The maximum nr. of bytes that can be allocated in one go.
    pub max_page_size: u64,
    /// The required memory offset alignment in bytes.
    pub alignment: u64,
}

/// Who a device is, and what its compiled code is keyed to.
///
/// The two fields answer different questions and must not be confused. `name`
/// is for people: it names the physical part, and two machines holding the same
/// part report the same name. `fingerprint` is for correctness: it is verbatim
/// the string this runtime passes to
/// [`compilation_store`](../../cubecl_runtime/compiler/fn.compilation_store.html),
/// which is what puts a compiled artifact out of reach of a machine that cannot
/// run it.
///
/// Reporting the fingerprint here rather than recomputing it is the whole
/// point: a backend derives it once and hands it to both consumers, so the
/// identity a bundle is stamped with and the namespace its kernels live under
/// cannot drift apart.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeviceIdentity {
    /// The device as it names itself — `AMD Radeon 8060S Graphics`,
    /// `NVIDIA H100 PCIe`. Distinct parts may share a name, so no capability
    /// and no compiled artifact may be keyed to it.
    pub name: String,
    /// What this runtime compiles *for* — `hip-kernel_gfx1151`, `ptx_sm90`.
    /// Verbatim the `compilation_store` fingerprint, so a namespace read back
    /// out of a bundle compares against it directly.
    pub fingerprint: String,
    /// The card behind the device, `None` for a CPU or a virtual device.
    pub physical: Option<PhysicalDevice>,
}

/// The card a device runs on, so one card reached through two runtimes (an
/// NVIDIA GPU under CUDA and under Vulkan) is recognized as one card.
///
/// `pci` is the key wherever the runtime reads it: every runtime that sees a
/// card on a bus reports the same address. `uuid` is the driver's own id and
/// NVIDIA reports one through CUDA and Vulkan; other vendors may not. The
/// ids and memory are what a placement needs to size a card.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PhysicalDevice {
    /// Where the card sits on the bus, when the runtime reports it.
    pub pci: Option<PciAddress>,
    /// The driver's id for the card, when the runtime reports it.
    pub uuid: Option<[u8; 16]>,
    /// PCI vendor id: `0x10de` NVIDIA, `0x1002` AMD, `0x8086` Intel.
    pub vendor_id: Option<u32>,
    /// PCI device id of the part, when the runtime reports it.
    pub device_id: Option<u32>,
    /// Bytes of memory on the card, when the runtime reports it.
    pub total_memory: Option<u64>,
}

/// A PCI address, spelled `0000:07:00.0` as the OS and every driver spell it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PciAddress {
    /// The PCI domain (segment).
    pub domain: u32,
    /// The bus number.
    pub bus: u8,
    /// The device number on the bus.
    pub device: u8,
    /// The function of the device.
    pub function: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// A PCI address that could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciAddressError(pub String);

impl fmt::Display for PciAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a PCI address: {}", self.0)
    }
}

impl FromStr for PciAddress {
    type Err = PciAddressError;

    /// Accepts `0000:07:00.0` and the domainless `07:00.0` CUDA also emits.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let err = || PciAddressError(String::from(text));
        let (rest, function) = text.rsplit_once('.').ok_or_else(err)?;
        let mut parts = rest.rsplitn(3, ':');
        let device = parts.next().ok_or_else(err)?;
        let bus = parts.next().ok_or_else(err)?;
        let domain = parts.next().unwrap_or("0");
        Ok(Self {
            domain: u32::from_str_radix(domain, 16).map_err(|_| err())?,
            bus: u8::from_str_radix(bus, 16).map_err(|_| err())?,
            device: u8::from_str_radix(device, 16).map_err(|_| err())?,
            function: u8::from_str_radix(function, 16).map_err(|_| err())?,
        })
    }
}

/// Properties of what the device can do, like what `Feature` are
/// supported by it and what its memory properties are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    /// The features supported by the runtime.
    pub features: Features,
    /// The memory properties of this client.
    pub memory: MemoryDeviceProperties,
    /// The topology properties of this client.
    pub hardware: HardwareProperties,
    /// The method used for profiling on the device.
    pub timing_method: TimingMethod,
    /// Who the device is, and what its kernels are keyed to.
    pub identity: DeviceIdentity,
}

impl TypeHash for DeviceProperties {
    fn write_hash(_hasher: &mut impl core::hash::Hasher) {
        // ignored.
    }
}

impl DeviceProperties {
    /// Create a new feature set with the given features and memory properties.
    ///
    /// `identity` is a required argument rather than something a backend may
    /// fill in afterwards, so a runtime cannot ship reporting an anonymous
    /// device — the failure mode that leaves a bundle unable to say what it was
    /// built for.
    pub fn new(
        features: Features,
        memory_props: MemoryDeviceProperties,
        hardware: HardwareProperties,
        timing_method: TimingMethod,
        identity: DeviceIdentity,
    ) -> Self {
        DeviceProperties {
            features,
            memory: memory_props,
            hardware,
            timing_method,
            identity,
        }
    }

    /// Get the usages for a type
    pub fn type_usage(&self, ty: ElemType) -> EnumSet<TypeUsage> {
        self.features.type_usage(ty)
    }

    /// Get the complex capability families for a type.
    pub fn complex_usage(&self, ty: ElemType) -> EnumSet<ComplexUsage> {
        self.features.complex_usage(ty)
    }

    /// Whether a complex type supports the requested capability family.
    pub fn supports_complex_usage(&self, ty: ElemType, usage: ComplexUsage) -> bool {
        self.features.supports_complex_usage(ty, usage)
    }

    /// Get the usages for an atomic type
    pub fn atomic_type_usage(&self, ty: Type) -> EnumSet<AtomicUsage> {
        self.features.atomic_type_usage(ty)
    }

    /// Whether the type is supported in any way
    pub fn supports_type(&self, ty: impl Into<Type>) -> bool {
        self.features.supports_type(ty)
    }

    /// Whether the address type is supported in any way
    pub fn supports_address(&self, ty: impl Into<AddressType>) -> bool {
        self.features.supports_address(ty)
    }

    /// Register an address type to the features
    pub fn register_address_type(&mut self, ty: impl Into<AddressType>) {
        self.features.types.address.insert(ty.into());
    }

    /// Register an address type to the features
    pub fn register_atomic_type_usage(&mut self, ty: Type, uses: impl Into<EnumSet<AtomicUsage>>) {
        *self.features.types.atomic.entry(ty).or_default() |= uses.into();
    }

    /// Register a storage type to the features
    pub fn register_type_usage(
        &mut self,
        ty: impl Into<ElemType>,
        uses: impl Into<EnumSet<TypeUsage>>,
    ) {
        *self.features.types.elem.entry(ty.into()).or_default() |= uses.into();
    }

    /// Register complex capability families for an element type.
    pub fn register_complex_usage(
        &mut self,
        ty: impl Into<ElemType>,
        uses: impl Into<EnumSet<ComplexUsage>>,
    ) {
        *self.features.types.complex.entry(ty.into()).or_default() |= uses.into();
    }

    /// Register a semantic type to the features
    pub fn register_semantic_type(&mut self, ty: SemanticType) {
        self.features.types.semantic.insert(ty);
    }

    /// Register an opaque type to the features
    pub fn register_opaque_type(&mut self, ty: OpaqueType) {
        self.features.types.opaque.insert(ty);
    }

    /// Create a stable hash of all device properties relevant to kernel compilation. Can be used
    /// as a stable checksum for a compilation cache.
    pub fn checksum(&self) -> u64 {
        let state = foldhash::fast::FixedState::default();
        let mut hasher = state.build_hasher();
        self.features.hash(&mut hasher);
        self.hardware.hash(&mut hasher);
        hasher.finish()
    }
}

/// Unchecked optimizations for float operations. May cause precision differences, or undefined
/// behaviour if the relevant conditions are not followed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, EnumSetType)]
pub enum FastMath {
    /// Assume values are never `NaN`. If they are, the result is considered undefined behaviour.
    NotNaN,
    /// Assume values are never `Inf`/`-Inf`. If they are, the result is considered undefined
    /// behaviour.
    NotInf,
    /// Ignore sign on zero values.
    UnsignedZero,
    /// Allow swapping float division with a reciprocal, even if that swap would change precision.
    AllowReciprocal,
    /// Allow contracting float operations into fewer operations, even if the precision could
    /// change.
    AllowContraction,
    /// Allow reassociation for float operations, even if the precision could change.
    AllowReassociation,
    /// Allow all mathematical transformations for float operations, including contraction and
    /// reassociation, even if the precision could change.
    AllowTransform,
    /// Allow using lower precision intrinsics
    ReducedPrecision,
}

impl FastMath {
    pub const fn all() -> EnumSet<FastMath> {
        EnumSet::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn a_pci_address_round_trips_and_defaults_its_domain() {
        let address = PciAddress {
            domain: 0,
            bus: 7,
            device: 0,
            function: 0,
        };
        assert_eq!(address.to_string(), "0000:07:00.0");
        assert_eq!("0000:07:00.0".parse::<PciAddress>(), Ok(address));
        assert_eq!("07:00.0".parse::<PciAddress>(), Ok(address));
        assert_eq!(
            "0001:a3:1f.7".parse::<PciAddress>(),
            Ok(PciAddress {
                domain: 1,
                bus: 0xa3,
                device: 0x1f,
                function: 7
            })
        );
        assert!("07:00".parse::<PciAddress>().is_err());
        assert!("gpu".parse::<PciAddress>().is_err());
    }
}
