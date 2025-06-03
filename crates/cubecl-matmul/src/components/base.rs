use cubecl_core::prelude::*;

use super::{InputRuntimeArg, MatmulConfigFactory, MatmulSpec, OutputRuntimeArg};

/// Provides launch entry point to solve a matmul
pub trait MatmulLaunch: MatmulConfigFactory {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: <Self as MatmulConfigFactory>::Config,
    );
}
