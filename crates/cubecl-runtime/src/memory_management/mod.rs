pub(crate) mod memory_pool;

mod base;

pub use base::*;

/// Dynamic memory management strategy.
mod memory_manage;
use cubecl_common::CubeDim;
pub use memory_manage::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The type of memory pool to use.
#[derive(Debug, Clone)]
pub enum PoolType {
    /// Use a memory where every allocation is a separate page.
    ExclusivePages {
        /// The minimum number of bytes to allocate in this pool.
        max_alloc_size: u64,
    },
    /// Use a memory where each allocation is a slice of a bigger allocation.
    SlicedPages {
        /// The page size to allocate.
        page_size: u64,
        /// The maximum size of a slice to allocate in the pool.
        max_slice_size: u64,
    },
}

/// Options to create a memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolOptions {
    /// What kind of pool to use.
    pub pool_type: PoolType,
    /// Period after which allocations are deemed unused and deallocated.
    ///
    /// This period is measured in the number of allocations in the parent allocator. If a page
    /// in the pool was unused for the entire period, it will be deallocated. This period is
    /// approximmate, as checks are only done occasionally.
    pub dealloc_period: Option<u64>,
}

/// High level configuration of memory management.
#[derive(Clone, Debug)]
pub enum MemoryConfiguration {
    /// The default preset, which uses pools that allocate sub slices.
    #[cfg(not(exclusive_memory_only))]
    SubSlices,
    /// Default preset for using exclusive pages.
    /// This can be necessary for backends don't support sub-slices.
    ExclusivePages,
    /// Custom settings.
    Custom {
        /// Options for each pool to construct. When allocating, the first
        /// possible pool will be picked for an allocation.
        pool_options: Vec<MemoryPoolOptions>,
    },
}

#[allow(clippy::derivable_impls)]
impl Default for MemoryConfiguration {
    fn default() -> Self {
        #[cfg(exclusive_memory_only)]
        {
            MemoryConfiguration::ExclusivePages
        }
        #[cfg(not(exclusive_memory_only))]
        {
            MemoryConfiguration::SubSlices
        }
    }
}

/// Properties of the device related to allocation.
#[derive(Debug, Clone)]
pub struct MemoryDeviceProperties {
    /// The maximum nr. of bytes that can be allocated in one go.
    pub max_page_size: u64,
    /// The required memory offset alignment in bytes.
    pub alignment: u64,
}

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
#[derive(Debug, Clone)]
pub struct HardwareProperties {
    /// The minimum size of a plane on this device
    pub plane_size_min: u32,
    /// The maximum size of a plane on this device
    pub plane_size_max: u32,
    /// minimum number of bindings for a kernel that can be used at once.
    pub max_bindings: u32,
    /// Maximum amount of shared memory, in bytes
    pub max_shared_memory_size: usize,
    /// Maximum `CubeCount` in x, y and z dimensions
    pub max_cube_count: CubeDim,
    /// Maximum number of total units in a cube
    pub max_units_per_cube: u32,
    /// Maximum `CubeDim` in x, y, and z dimensions
    pub max_cube_dim: CubeDim,
    /// Number of streaming multiprocessors (SM), if available
    pub num_streaming_multiprocessors: Option<u32>,
    /// Number of tensor cores per SM, if any
    pub num_tensor_cores: Option<u32>,
    /// The minimum tiling dimension for a single axis in tensor cores.
    ///
    /// For a backend that only supports 16x16x16, the value would be 16.
    /// For a backend that also supports 32x8x16, the value would be 8.
    pub min_tensor_cores_dim: Option<u32>,
}

impl HardwareProperties {
    /// Plane size that is defined for the device.
    pub fn defined_plane_size(&self) -> Option<u32> {
        if self.plane_size_min == self.plane_size_max {
            Some(self.plane_size_min)
        } else if self.plane_size_min == 32 {
            // Normally 32 is chosen by default when it's the min plane size.
            Some(self.plane_size_min)
        } else {
            None
        }
    }
}
