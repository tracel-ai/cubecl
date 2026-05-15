use crate::{AddressType, SemanticType, StorageType, Type};
use alloc::collections::{BTreeMap, BTreeSet};

use enumset::EnumSetType;

pub use enumset::EnumSet;

/// Features supported by a runtime
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct Features {
    /// Plane features supported by this runtime.
    pub plane: EnumSet<Plane>,
    /// Clustered launches and intra-cluster operations like cluster shared memory
    pub cube_cluster: bool,
    /// Enables changing the type of containers during kernel execution.
    pub memory_reinterpret: bool,
    /// Enables explicit alignment. If false, alignment still compiles, but isn't actually applied.
    pub alignment: bool,

    /// Type support
    pub types: Types,
    /// Matrix multiplication features
    pub matmul: MatmulFeatures,

    /// Whether `copy_async` is supported
    pub copy_async: bool,
    /// Tensor Memory Accelerator supported features
    pub tma: EnumSet<Tma>,
    /// Whether vectors can be read from / stored to addresses not aligned
    /// with the `vector_size`
    pub unaligned_io: bool,
}

/// Type support for a device
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct Types {
    /// Valid address types
    pub address: BTreeSet<AddressType>,
    /// Types supported by this runtime, and which usages they support.
    pub storage: BTreeMap<StorageType, EnumSet<TypeUsage>>,
    /// Complex-specific capability families supported by this runtime.
    pub complex: BTreeMap<StorageType, EnumSet<ComplexUsage>>,
    /// Semantic constructs supported by this runtime.
    pub semantic: BTreeSet<SemanticType>,
    /// Supported vector types for atomic ops, only specific vectorizations for specific types are
    /// supported here. Not all vector types are supported as scalars, i.e. Vulkan on Nvidia only
    /// supports vectorized `f16`, not scalar. Only use the exact vectorizations registered here.
    /// These may not be supported everywhere - in practice, f32 vectors are only supported in global
    /// memory.
    pub atomic: BTreeMap<Type, EnumSet<AtomicUsage>>,
}

/// Matrix multiplication-related features
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct MatmulFeatures {
    /// The cmma feature enables cooperative matrix-multiply and accumulate operations.
    pub cmma: BTreeSet<MmaConfig>,
    /// The manual MMA feature enables cooperative matrix-multiply with manually managed data
    /// movement
    pub mma: BTreeSet<MmaConfig>,
    /// Scaled MMA allows combining matrix multiplication with unscaling quantized values into a single
    /// instruction. Scales must fit a specific layout and block size.
    pub scaled_mma: BTreeSet<ScaledMmaConfig>,
    /// Types supported for ldmatrix, if any
    pub ldmatrix: BTreeSet<StorageType>,
    /// Types supported by stmatrix, if any
    pub stmatrix: BTreeSet<StorageType>,
}

/// Operations allowed for this type. CMMA is defined separately.
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum TypeUsage {
    /// Conversion to/from the type. All types should support this.
    Conversion,
    /// All math/logic instructions except dot product
    Arithmetic,
    /// Dot product, mainly for BF16 on Intel
    DotProduct,
    /// Whether this type can be stored in a buffer
    Buffer,
}

/// Complex capability families allowed for a complex storage type.
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum ComplexUsage {
    /// Core ML-centric complex functionality: arithmetic, negation, conjugation, real/imag.
    Core,
    /// Equality and inequality comparisons.
    Compare,
    /// Higher-level math functions such as exp/log/sin/cos/sqrt/tanh/powf and abs.
    Math,
}

impl TypeUsage {
    pub fn all() -> EnumSet<Self> {
        EnumSet::all()
    }

    pub fn no_store() -> EnumSet<Self> {
        TypeUsage::Conversion | TypeUsage::Arithmetic
    }

    pub fn maybe_store(storable: bool) -> EnumSet<Self> {
        if storable {
            EnumSet::all()
        } else {
            Self::no_store()
        }
    }
}

/// Atomic operations allowed for this type.
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum AtomicUsage {
    /// Atomic loads and stores
    LoadStore,
    /// Atomic add/sub
    Add,
    /// Atomic min/max
    MinMax,
}

impl AtomicUsage {
    pub fn all() -> EnumSet<Self> {
        EnumSet::all()
    }
}

/// Supported plane features
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum Plane {
    /// Basic plane-wide operations
    Ops,
    /// Plane-wide sync
    Sync,
    /// Allows using plane operations with divergent control flow.
    NonUniformControlFlow,
}

/// Shape and element types of a valid MMA configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MmaConfig {
    /// Element of the A matrix
    pub a_type: StorageType,
    /// Element of the B matrix
    pub b_type: StorageType,
    /// Element of the C/D matrices
    pub cd_type: StorageType,
    /// The size of the matrix on the `m` dimension
    pub m: u32,
    /// The size of the matrix on the `n` dimension
    pub n: u32,
    /// The size of the matrix on the `k` dimension
    pub k: u32,
}

/// Shape and element types of a valid block-scaled MMA configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScaledMmaConfig {
    /// Element of the A matrix
    pub a_type: StorageType,
    /// Element of the B matrix
    pub b_type: StorageType,
    /// Element of the C/D matrices
    pub cd_type: StorageType,
    /// Element of the blocks scales
    pub scales_type: StorageType,
    /// The size of the matrix on the `m` dimension
    pub m: u32,
    /// The size of the matrix on the `n` dimension
    pub n: u32,
    /// The size of the matrix on the `k` dimension
    pub k: u32,
    /// Number of scales per tile row/col.
    /// A scale factor of 2 means `m x 2` scales for A and `2 x n` for B (in CUDA)
    /// Scales blocks must be organized along the natural `vector_layout` of the operation
    pub scales_factor: u32,
}

/// Atomic features that may be supported by a ``Runtime``.
#[derive(Debug, PartialOrd, Ord, EnumSetType)]
pub enum Tma {
    /// Base feature set for tensor memory accelerator features. Includes tiling and im2col
    Base,
    /// im2colWide encoding for tensor map.
    Im2colWide,
    /// Different atomicities for 128-byte swizzle, i.e. 128-byte with 32-byte atomicity.
    SwizzleAtomicity,
}

impl Features {
    /// Get the usages for a type
    pub fn type_usage(&self, ty: StorageType) -> EnumSet<TypeUsage> {
        self.types
            .storage
            .get(&ty)
            .cloned()
            .unwrap_or_else(EnumSet::empty)
    }

    /// Get the complex capability families for a type.
    pub fn complex_usage(&self, ty: StorageType) -> EnumSet<ComplexUsage> {
        self.types
            .complex
            .get(&ty)
            .cloned()
            .unwrap_or_else(EnumSet::empty)
    }

    /// Get the usages for an atomic type
    pub fn atomic_type_usage(&self, ty: Type) -> EnumSet<AtomicUsage> {
        self.types
            .atomic
            .get(&ty)
            .cloned()
            .unwrap_or_else(EnumSet::empty)
    }

    /// Whether the type is supported in any way
    pub fn supports_type(&self, ty: impl Into<Type>) -> bool {
        match ty.into() {
            Type::Scalar(storage_type) | Type::Vector(storage_type, _) => {
                self.types.storage.contains_key(&storage_type)
            }
            Type::Semantic(semantic_type) => self.types.semantic.contains(&semantic_type),
        }
    }

    /// Whether the address type is supported in any way
    pub fn supports_address(&self, ty: impl Into<AddressType>) -> bool {
        self.types.address.contains(&ty.into())
    }

    /// Whether a complex storage type supports the requested capability family.
    pub fn supports_complex_usage(&self, ty: impl Into<StorageType>, usage: ComplexUsage) -> bool {
        self.complex_usage(ty.into()).contains(usage)
    }
}
