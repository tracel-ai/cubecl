use crate::codegen::Compiler;
use crate::compute::CubeTask;
use cubecl_common::device::Device;
use cubecl_ir::{StorageType, TargetProperties};
use cubecl_runtime::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

pub use cubecl_runtime::channel;
pub use cubecl_runtime::client;
pub use cubecl_runtime::server;
pub use cubecl_runtime::tune;

/// Max width of loads. May want to make this a property in the future, since Nvidia seems have some
/// support for 256-bit loads on Blackwell.
const LOAD_WIDTH: usize = 128;

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

    /// The maximum line size that can be used for global buffer bindings.
    ///
    /// # Notes
    ///
    /// Normally any line size is supported within kernels (when using registers), but for global
    /// line size it's not the case.
    fn max_global_line_size() -> u8 {
        u8::MAX
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    fn io_optimized_line_sizes(elem: &StorageType) -> impl Iterator<Item = u8> + Clone {
        let max = (LOAD_WIDTH / elem.size_bits()) as u8;
        let supported = Self::supported_line_sizes();
        supported.iter().filter(move |v| **v <= max).cloned()
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    /// Ignores native support, and allows all line sizes. This means the returned size may be
    /// unrolled, and may not support dynamic indexing.
    fn io_optimized_line_sizes_unchecked(elem: &StorageType) -> impl Iterator<Item = u8> + Clone {
        let max = LOAD_WIDTH / elem.size_bits();
        let max = usize::min(Self::max_global_line_size() as usize, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = f32::log2(max as f32) as u32 + 1;

        (0..num_candidates).into_iter().map(|i| 2u8.pow(i)).rev()
    }

    /// Returns the maximum cube count on each dimension that can be launched.
    fn max_cube_count() -> (u32, u32, u32);

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool;

    /// Returns the properties of the target hardware architecture.
    fn target_properties() -> TargetProperties;
}
