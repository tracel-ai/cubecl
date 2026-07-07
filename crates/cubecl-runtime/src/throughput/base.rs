use crate::{client::ComputeClient, runtime::Runtime, throughput::ThroughputKey};
use alloc::boxed::Box;

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// The executable kernel closure to be evaluated.
    pub kernel: Box<dyn Fn(usize)>,
    /// The number of operations processed in one iteration.
    pub ops_count: usize,
}

/// Hardware execution parameters for launching a compute kernel.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// The number of threads per cube.
    pub cube_dim: usize,
    /// The total number of cubes to dispatch.
    pub cube_count: usize,
    /// The vectorization factor (e.g., 4 for `vec4` operations).
    pub vector_size: usize,
    /// The number of threads in a hardware execution plane.
    pub plane_size: usize,
}

/// A trait for running throughput benchmarks on compute kernels.
pub trait ThroughputRunner<R: Runtime> {
    /// Builds a kernel configuration for the given client, dtype, and launch config.
    fn build_kernel(
        client: &ComputeClient<R>,
        key: ThroughputKey,
        config: LaunchConfig,
    ) -> KernelConfig;
}
