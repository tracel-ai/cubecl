use cubecl_core::prelude::*;

use crate::matmul::kernels::{matmul::AdvancedConfig, MatmulAvailabilityError};

use super::{config::MatmulConfig, MatmulProblem};

/// Provides configuration for a matmul kernel at any level
pub trait MatmulKernel<I: Numeric, O: Numeric> {
    /// Configuration tailored to the matmul implementation
    type Config: MatmulConfig;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: Self::Config);

    /// Checks if the client can handle the features used in this computation
    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError>;

    /// Create config for this matmul, given launch information
    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config;
}

/// Provides launch entry point to solve a matmul
pub trait MatmulLaunch<I: Numeric, O: Numeric>: MatmulKernel<I, O> {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as MatmulKernel<I, O>>::Config,
    );
}
