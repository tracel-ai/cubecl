use crate::{
    matmul::{
        components::{
            CompleteStageTiling, InputIdent, InvalidConfigError, MatmulPrecision, MatmulSelection,
            global::args::MatmulArgs,
            stage::{StageBuffering, StageMatmulFamily, StageVectorization},
            tile::TileMatmulFamily,
        },
        kernels::MatmulAvailabilityError,
    },
    tensor::TensorHandle,
};
use cubecl_core::prelude::*;

use super::base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem};

pub mod simple;
pub mod simple_tma;

pub type StageInput = (CompleteStageTiling, StageBuffering, StageVectorization);

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily<Input = StageInput>;
    type GlobalConvolution: ConvolutionFamily<Input = StageInput>;

    type Args: MatmulArgs;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim;
    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount;

    /// Make a convolution config from a convolution problem, and launch options
    fn make_config(
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        let config = Self::GlobalConvolution::make_config(input, problem, cube_dim, cube_count);
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
