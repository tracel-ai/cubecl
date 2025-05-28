use crate::{
    matmul::{
        components::{
            InputIdent, InvalidConfigError, MatmulLineSizes, MatmulPrecision,
            global::{args::MatmulArgs, load::LoaderMode},
            stage::{NumStages, StageBuffering, StageMatmulFamily, TilesPerPartition},
            tile::TileMatmulFamily,
        },
        kernels::{
            MatmulAvailabilityError,
            matmul::{
                GlobalInput, LoadingPrecomputeStrategy, MatmulSelection, MultiRowStrategy,
                StageInput,
            },
        },
    },
    tensor::TensorHandle,
};
use cubecl_core::{ir::Elem, prelude::*};

use super::base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem};

pub mod multi_stage_tma;
pub mod simple;
pub mod simple_tma;

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily<Input = StageInput>;
    type GlobalConvolution: ConvolutionFamily<Input = GlobalInput<StageInput>>;
    type MatmulSelection: MatmulSelection;

    type Args: MatmulArgs;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim;
    fn cube_count(selection: &Self::MatmulSelection, problem: &ConvolutionProblem) -> CubeCount;
    fn num_stages() -> NumStages;

    // TODO delete in favor of using selection directly
    fn partition(selection: &Self::MatmulSelection) -> TilesPerPartition {
        selection.tiles_per_partition()
    }

    fn multi_row_strategy() -> MultiRowStrategy {
        MultiRowStrategy::Never
    }

    fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }

    fn loader_mode() -> LoaderMode {
        LoaderMode::Relaxed
    }

    fn stage_buffering_strategy() -> StageBuffering {
        StageBuffering::Double
    }

    /// Make a convolution config from a convolution problem, and launch options
    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        let config = Self::GlobalConvolution::make_config::<R, MP>(
            client, input, problem, line_sizes, cube_dim, cube_count,
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

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> Self::MatmulSelection;

    fn line_sizes(
        problem: &ConvolutionProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
    ) -> MatmulLineSizes {
        MatmulLineSizes::new_maximized(&problem.as_matmul_problem(), in_available, out_available)
    }
}
