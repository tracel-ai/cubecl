use alloc::boxed::Box;
use cubecl_common::device::Device;
use cubecl_ir::{LineSize, TargetProperties};

use crate::{
    client::ComputeClient,
    compiler::{Compiler, CubeTask},
    server::ComputeServer,
};

/// Runtime for the `CubeCL`.
pub trait Runtime: Sized + Send + Sync + 'static + core::fmt::Debug {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Box<dyn CubeTask<Self::Compiler>>>;
    /// The device used to retrieve the compute client.
    type Device: Device;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self>;

    /// The runtime name on the given device.
    fn name(client: &ComputeClient<Self>) -> &'static str;

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }

    /// Returns the supported line sizes for the current runtime's compiler.
    fn supported_line_sizes() -> &'static [LineSize];

    /// The maximum line size that can be used for global buffer bindings.
    fn max_global_line_size() -> LineSize {
        u8::MAX as usize
    }

    /// Returns the maximum cube count on each dimension that can be launched.
    fn max_cube_count() -> (u32, u32, u32);

    /// Whether a tensor with `shape` and `strides` can be read as is. If the result is false, the
    /// tensor should be made contiguous before reading.
    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool;

    /// Returns the properties of the target hardware architecture.
    fn target_properties() -> TargetProperties;
}
