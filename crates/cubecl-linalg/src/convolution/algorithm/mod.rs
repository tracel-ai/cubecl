use crate::{
    matmul::{
        components::{
            CompleteStageTiling, InputIdent, InvalidConfigError, MatmulPrecision, MatmulSelection,
            global::args::MatmulArgs,
            stage::{StageBuffering, StageMatmulFamily, StageVectorization},
            tile::TileMatmulFamily,
        },
        kernels::{
            MatmulAvailabilityError,
            matmul::{LoadingPrecomputeStrategy, MultiRowStrategy},
        },
    },
    tensor::TensorHandle,
};
use cubecl_core::prelude::*;

use super::base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem};

pub mod multi_stage_tma;
pub mod simple;
pub mod simple_tma;

pub type GlobalInput = (StageInput, LoadingPrecomputeStrategy);
pub type StageInput = (
    CompleteStageTiling,
    StageBuffering,
    StageVectorization,
    (u32, u32),
);

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily<Input = StageInput>;
    type GlobalConvolution: ConvolutionFamily<Input = GlobalInput>;

    type Args: MatmulArgs;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim;
    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount;
    fn num_stages() -> (u32, u32);

    fn multi_row_strategy() -> MultiRowStrategy {
        MultiRowStrategy::Never
    }

    fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }

    fn stage_buffering_strategy() -> StageBuffering {
        StageBuffering::Double
    }

    /// Make a convolution config from a convolution problem, and launch options
    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        let config = Self::GlobalConvolution::make_config::<R, MP>(
            client, input, problem, cube_dim, cube_count,
        );
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        <Self::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
            client, config,
        )
    }

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: InputIdent,
    ) -> TensorHandle<R, E>;
}
