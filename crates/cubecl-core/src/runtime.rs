use crate::DeviceId;
use crate::{codegen::Compiler, compute::CubeTask, ir::Elem};
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
    type Device: Default + Clone + core::fmt::Debug + Send + Sync;

    /// Fetch the id for the given device.
    fn device_id(device: &Self::Device) -> DeviceId;

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
    fn line_size_elem(elem: &Elem) -> impl Iterator<Item = u8> + Clone {
        Self::supported_line_sizes()
            .iter()
            .filter(|v| **v as usize * elem.size() <= 16)
            .cloned() // 128 bits
    }

    /// Returns the maximum cube count on each dimension that can be launched.
    fn max_cube_count() -> (u32, u32, u32);

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool;
}

/// Every feature that can be supported by a [cube runtime](Runtime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    /// The plane feature enables all basic warp/subgroup operations.
    Plane,
    /// The cmma feature enables cooperative matrix-multiply and accumulate operations.
    Cmma {
        a: Elem,
        b: Elem,
        c: Elem,
        m: u8,
        k: u8,
        n: u8,
    },
    CmmaWarpSize(i32),
    Type(Elem),
    /// Features supported for floating point atomics. For integers, all methods are supported as
    /// long as the type is.
    AtomicFloat(AtomicFeature),
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
