use alloc::string::String;
use core::{
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    str::FromStr,
};

use crate::{
    AddressType, ElemType, EnumSet, EnumSetType, OpaqueType, SemanticType, Type, TypeHash,
    VectorSize,
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
/// `name` and `fingerprint` answer different questions and must not be confused. `name`
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
    /// The card behind the device, `None` for a device that is no card: a CPU, or a software
    /// rasterizer.
    pub physical: Option<PhysicalDevice>,
}

/// The card a device runs on. Two runtimes report different fields for one card, so compare with
/// [`is_same_card`](Self::is_same_card), not `==`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct PhysicalDevice {
    /// The key every runtime reports alike on Linux.
    pub pci_address: Option<PciAddress>,
    /// The key every runtime reports alike on Windows.
    pub luid: Option<AdapterLuid>,
    pub vendor: Option<PciVendor>,
}

impl PhysicalDevice {
    /// Whether `other` is known to be this card through another runtime: by PCI address, otherwise
    /// by LUID. A card with neither matches nothing, itself included, since two such cards of one
    /// make compare equal.
    pub fn is_same_card(&self, other: &Self) -> bool {
        if let (Some(mine), Some(theirs)) = (self.pci_address, other.pci_address) {
            return mine == theirs;
        }
        matches!((self.luid, other.luid), (Some(mine), Some(theirs)) if mine == theirs)
    }
}

/// The id Windows gives a graphics adapter. It changes on restart, so it has no serialization or
/// text form: a stored key wants [`PhysicalDevice::pci_address`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdapterLuid([u8; 8]);

impl AdapterLuid {
    /// From the eight bytes of a Windows `LUID`, low part first.
    pub fn new(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    pub fn from_parts(low_part: u32, high_part: i32) -> Self {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&low_part.to_le_bytes());
        bytes[4..].copy_from_slice(&high_part.to_le_bytes());
        Self(bytes)
    }

    pub fn bytes(self) -> [u8; 8] {
        self.0
    }
}

/// The maker of a card, by PCI vendor id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PciVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    /// Mali GPUs.
    Arm,
    /// Adreno GPUs.
    Qualcomm,
    Other(u32),
}

impl PciVendor {
    pub fn id(self) -> u32 {
        match self {
            Self::Nvidia => 0x10de,
            Self::Amd => 0x1002,
            Self::Intel => 0x8086,
            Self::Apple => 0x106b,
            Self::Arm => 0x13b5,
            Self::Qualcomm => 0x5143,
            Self::Other(id) => id,
        }
    }
}

impl From<u32> for PciVendor {
    fn from(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106b => Self::Apple,
            0x13b5 => Self::Arm,
            0x5143 => Self::Qualcomm,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for PciVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvidia => f.write_str("NVIDIA"),
            Self::Amd => f.write_str("AMD"),
            Self::Intel => f.write_str("Intel"),
            Self::Apple => f.write_str("Apple"),
            Self::Arm => f.write_str("Arm"),
            Self::Qualcomm => f.write_str("Qualcomm"),
            Self::Other(id) => write!(f, "{id:#06x}"),
        }
    }
}

/// `domain:bus:device.function`, written `0000:07:00.0`. CUDA and NVML call it the bus id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u8,
    pub device: u8,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciAddressError(pub String);

impl fmt::Display for PciAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a PCI address: {}", self.0)
    }
}

impl core::error::Error for PciAddressError {}

impl FromStr for PciAddress {
    type Err = PciAddressError;

    /// Also accepts the domainless `07:00.0` CUDA emits.
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
    fn a_vendor_keeps_its_id_whether_named_or_not() {
        for id in [0x10de, 0x1002, 0x8086, 0x106b, 0x13b5, 0x5143, 0x1af4] {
            assert_eq!(PciVendor::from(id).id(), id);
        }
        assert_eq!(PciVendor::from(0x10de), PciVendor::Nvidia);
        assert_eq!(PciVendor::from(0x1af4), PciVendor::Other(0x1af4));
        assert_eq!(PciVendor::Other(0x1af4).to_string(), "0x1af4");
    }

    #[test]
    fn a_luid_from_parts_is_the_bytes_vulkan_reports() {
        let bytes = [0x8a, 0x1d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            AdapterLuid::from_parts(0x0001_1d8a, 0),
            AdapterLuid::new(bytes)
        );
        assert_eq!(
            AdapterLuid::from_parts(1, -1).bytes(),
            [1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn a_card_is_matched_by_address_then_by_luid() {
        let address = |bus| {
            Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: 0,
            })
        };
        let luid = |low| Some(AdapterLuid::from_parts(low, 0));
        let card = |pci_address, luid| PhysicalDevice {
            pci_address,
            luid,
            vendor: None,
        };

        assert!(card(address(7), None).is_same_card(&card(address(7), luid(1))));
        assert!(!card(address(7), luid(1)).is_same_card(&card(address(8), luid(1))));
        assert!(card(None, luid(1)).is_same_card(&card(address(7), luid(1))));
        assert!(!card(None, luid(1)).is_same_card(&card(None, luid(2))));
        assert!(!card(None, None).is_same_card(&card(None, None)));
    }

    #[test]
    fn a_pci_address_round_trips_and_defaults_its_domain() {
        let id = PciAddress {
            domain: 0,
            bus: 7,
            device: 0,
            function: 0,
        };
        assert_eq!(id.to_string(), "0000:07:00.0");
        assert_eq!("0000:07:00.0".parse::<PciAddress>(), Ok(id));
        assert_eq!("07:00.0".parse::<PciAddress>(), Ok(id));
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
