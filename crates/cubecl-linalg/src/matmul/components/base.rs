use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{InputRuntimeArg, MatmulPrecision, MatmulSpec, OutputRuntimeArg};
use crate::matmul::kernels::{matmul::AdvancedConfig, MatmulAvailabilityError};

use super::{config::MatmulConfig, MatmulProblem};

/// Provides configuration for a matmul kernel at any level
pub trait MatmulConfigFactory: Send + Sync + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: MatmulConfig;
    type Input;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: Self::Config);

    /// Checks if the client can handle the features used in this computation
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        _client: &ComputeClient<R::Server, R::Channel>,
        _config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError>;

    /// Create config for this matmul, given launch information
    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config;
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MatmulSize {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct MatmulSelection {
    pub tile: MatmulSize,
    pub num_stagess: MatmulSize,
    pub plane_dim: u32,
}

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
