use crate::{
    Features, TypeUsage,
    memory_management::{HardwareProperties, MemoryDeviceProperties},
};
use cubecl_common::profile::TimingMethod;
use cubecl_ir::{SemanticType, StorageType, Type};
use enumset::EnumSet;

/// Properties of what the device can do, like what `Feature` are
/// supported by it and what its memory properties are.
use crate::server::AllocationKind;

/// Properties/features exposed by a device/runtime, used by higher layers for
/// capability checks and defaults.
#[derive(Debug)]
pub struct DeviceProperties {
    /// The features supported by the runtime.
    pub features: Features,
    /// The memory properties of this client.
    pub memory: MemoryDeviceProperties,
    /// The topology properties of this client.
    pub hardware: HardwareProperties,
    /// The method used for profiling on the device.
    pub timing_method: TimingMethod,
    /// Default allocation preference for rank > 1 tensors when both contiguous and
    /// inner‑contiguous row layouts are supported by the backend IO path.
    ///
    /// Backends can set this to `AllocationKind::Optimized` (pitched rows) when
    /// strided IO is efficient in hardware (e.g., CUDA/HIP), or to
    /// `AllocationKind::Contiguous` when contiguous copies are generally faster
    /// (e.g., WGPU/CPU by default).
    pub default_alloc_rank_gt1: AllocationKind,
}

impl DeviceProperties {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(
        features: Features,
        memory_props: MemoryDeviceProperties,
        hardware: HardwareProperties,
        timing_method: TimingMethod,
        default_alloc_rank_gt1: AllocationKind,
    ) -> Self {
        DeviceProperties {
            features,
            memory: memory_props,
            hardware,
            timing_method,
            default_alloc_rank_gt1,
        }
    }

    /// Get the usages for a type
    pub fn type_usage(&self, ty: StorageType) -> EnumSet<TypeUsage> {
        self.features.type_usage(ty)
    }

    /// Whether the type is supported in any way
    pub fn supports_type(&self, ty: impl Into<Type>) -> bool {
        self.features.supports_type(ty)
    }

    /// Register a storage type to the features
    pub fn register_type_usage(
        &mut self,
        ty: impl Into<StorageType>,
        uses: impl Into<EnumSet<TypeUsage>>,
    ) {
        *self.features.storage_types.entry(ty.into()).or_default() |= uses.into();
    }

    /// Register a semantic type to the features
    pub fn register_semantic_type(&mut self, ty: SemanticType) {
        self.features.semantic_types.insert(ty);
    }
}
