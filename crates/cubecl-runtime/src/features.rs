use alloc::collections::{BTreeMap, BTreeSet};

use cubecl_ir::{SemanticType, StorageType, Type};
use enumset::EnumSetType;

pub use enumset::EnumSet;

/// Features supported by a runtime
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Features {
    /// Plane features supported by this runtime.
    pub plane: EnumSet<Plane>,
    /// Clustered launches and intra-cluster operations like cluster shared memory
    pub cube_cluster: bool,
    /// Enables to change the line size of containers during kernel execution.
    pub dynamic_line_size: bool,
    /// Enables explicit alignment. If false, alignment still compiles, but isn't actually applied.
    pub alignment: bool,

    /// Types supported by this runtime, and which usages they support.
    pub storage_types: BTreeMap<StorageType, EnumSet<TypeUsage>>,
    /// Semantic constructs supported by this runtime.
    pub semantic_types: BTreeSet<SemanticType>,

    /// Whether `copy_async` is supported
    pub copy_async: bool,
    /// Tensor Memory Accelerator supported features
    pub tma: EnumSet<Tma>,
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
    /// Whether Lines can be read from / stored to addresses not aligned
    /// with the line_size
    pub unaligned_io: bool,
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
    /// Atomic loads and stores
    AtomicLoadStore,
    /// Atomic add/sub
    AtomicAdd,
    /// Atomic min/max
    AtomicMinMax,
}

/// Supported plane features
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum Plane {
    /// Basic plane-wide operations
    Ops,
    /// Plane-wide sync
    Sync,
}

/// Shape and element types of a valid MMA configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    /// Scales blocks must be organized along the natural `line_layout` of the operation
    pub scales_factor: u32,
}

/// Atomic features that may be supported by a [cube runtime](Runtime).
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
        self.storage_types
            .get(&ty)
            .cloned()
            .unwrap_or_else(EnumSet::empty)
    }

    /// Whether the type is supported in any way
    pub fn supports_type(&self, ty: impl Into<Type>) -> bool {
        match ty.into() {
            Type::Scalar(storage_type) | Type::Line(storage_type, _) => {
                self.storage_types.contains_key(&storage_type)
            }
            Type::Semantic(semantic_type) => self.semantic_types.contains(&semantic_type),
        }
    }
}

impl TypeUsage {
    /// All uses except atomics
    pub fn all_scalar() -> EnumSet<TypeUsage> {
        TypeUsage::Conversion | TypeUsage::Arithmetic | TypeUsage::DotProduct | TypeUsage::Buffer
    }

    /// All atomic uses
    pub fn all_atomic() -> EnumSet<TypeUsage> {
        TypeUsage::AtomicAdd | TypeUsage::AtomicLoadStore | TypeUsage::AtomicMinMax
    }
}
