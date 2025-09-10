use crate::codegen::Compiler;
use crate::compute::CubeTask;
use cubecl_ir::{StorageType, TargetProperties};
use cubecl_runtime::id::DeviceId;
use cubecl_runtime::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

pub use cubecl_runtime::channel;
pub use cubecl_runtime::client;
pub use cubecl_runtime::server;
pub use cubecl_runtime::tune;

/// Runtime for the CubeCL.
pub trait Runtime: Send + Sync + 'static + core::fmt::Debug {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Box<dyn CubeTask<Self::Compiler>>, Feature = Feature>;
    /// The channel used to communicate with the compute server.
    type Channel: ComputeChannel<Self::Server>;
    /// The device used to retrieve the compute client.
    type Device: Device;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel>;

    /// The runtime name on the given device.
    fn name(client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str;

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }

    /// Returns the supported line sizes for the current runtime's compiler.
    fn supported_line_sizes() -> &'static [u8];

    /// Returns all line sizes that are useful to perform IO operation on the given element.
    fn line_size_type(elem: &StorageType) -> impl Iterator<Item = u8> + Clone {
        Self::supported_line_sizes()
            .iter()
            .filter(|v| **v as usize * elem.size() <= 16)
            .cloned() // 128 bits
    }

    /// Returns the maximum cube count on each dimension that can be launched.
    fn max_cube_count() -> (u32, u32, u32);

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool;

    /// Returns the properties of the target hardware architecture.
    fn target_properties() -> TargetProperties;
}

/// Device trait for all cubecl devices.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync {
    /// Create a device from its [id](DeviceId).
    fn from_id(device_id: DeviceId) -> Self;
    /// Retrieve the [device id](DeviceId) from the device.
    fn to_id(&self) -> DeviceId;
    /// Returns the number of devices available under the provided type id.
    fn device_count(type_id: u16) -> usize;
    /// Returns the total number of devices that can be handled by the runtime.
    fn device_count_total() -> usize {
        Self::device_count(0)
    }
}

/// Every feature that can be supported by a [cube runtime](Runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    /// The plane feature enables all basic warp/subgroup operations.
    Plane,
    /// The cmma feature enables cooperative matrix-multiply and accumulate operations.
    Cmma {
        a: StorageType,
        b: StorageType,
        c: StorageType,
        m: u8,
        k: u8,
        n: u8,
    },
    /// The manual MMA feature enables cooperative matrix-multiply with manually managed data
    /// movement
    ManualMma {
        /// Element of the A matrix
        a_type: StorageType,
        /// Element of the B matrix
        b_type: StorageType,
        /// Element of the C/D matrices
        cd_type: StorageType,
        m: u32,
        n: u32,
        k: u32,
    },
    /// Scaled MMA allows combining matrix multiplication with unscaling quantized values into a single
    /// instruction. Scales must fit a specific layout and block size.
    ScaledMma {
        /// Element of the quantized A matrix
        a_type: StorageType,
        /// Element of the quantized B matrix
        b_type: StorageType,
        /// Element of the unquantized C/D matrices
        cd_type: StorageType,
        /// Element of the blocks scales
        scales_type: StorageType,
        m: u32,
        n: u32,
        k: u32,
        /// Number of scales per tile row/col.
        /// A scale factor of 2 means `m x 2` scales for A and `2 x n` for B (in CUDA)
        /// Scales blocks must be organized along the natural `line_layout` of the operation
        scales_factor: u32,
    },
    CmmaWarpSize(i32),
    Type(StorageType),
    /// Features supported for floating point atomics.
    AtomicFloat(AtomicFeature),
    /// Features supported for integer atomics.
    AtomicInt(AtomicFeature),
    /// Features supported for unsigned integer atomics.
    AtomicUInt(AtomicFeature),
    /// The pipeline feature enables pipelined (async) operations
    Pipeline,
    /// The barrier feature enables barrier (async) operations
    Barrier,
    /// Tensor Memory Accelerator features. Minimum H100/RTX 5000 series for base set
    Tma(TmaFeature),
    /// Clustered launches and intra-cluster operations like cluster shared memory
    CubeCluster,
    /// Enables to change the line size of containers during kernel execution.
    DynamicLineSize,
    /// Enables synchronization within a plane only
    SyncPlane,
    /// Enables using plane-wide operations like plane_sum, etc.
    PlaneOps,
}

/// Atomic features that may be supported by a [cube runtime](Runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtomicFeature {
    LoadStore,
    Add,
    MinMax,
}

/// Atomic features that may be supported by a [cube runtime](Runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TmaFeature {
    /// Base feature set for tensor memory accelerator features. Includes tiling and im2col
    Base,
    /// im2colWide encoding for tensor map.
    /// TODO: Not yet implemented due to missing `cudarc` support
    Im2colWide,
}
