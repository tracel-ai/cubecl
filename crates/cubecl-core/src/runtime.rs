use crate::codegen::Compiler;
use crate::compute::CubeTask;
use cubecl_common::device::Device;
use cubecl_ir::{StorageType, TargetProperties};
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
    type Server: ComputeServer<Kernel = Box<dyn CubeTask<Self::Compiler>>>;
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
