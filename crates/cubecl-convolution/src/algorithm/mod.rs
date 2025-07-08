use cubecl_matmul::{
    components::{
        AvailableLineSizes, InputIdent, MatmulLineSizes, MatmulPrecision, MatmulSetupError,
        global::{LoadSpecializationConfig, args::MatmulArgs, load::LoaderMode},
        stage::{NumStages, PartitionBuffering, StageMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::layered::{LoadingPrecomputeStrategy, MatmulSelection, MultiRowStrategy},
};

use cubecl_std::tensor::TensorHandle;

use cubecl_core::{ir::Elem, prelude::*};

use super::base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem};

pub mod multi_stage_tma;
pub mod simple;
pub mod simple_tma;

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalConvolution: ConvolutionFamily;

    type Args: MatmulArgs;

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_in_stage_m();
        let n_stage = selection.tiling_scheme.elements_in_stage_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn num_stages() -> NumStages;

    fn multi_row_strategy() -> MultiRowStrategy {
        MultiRowStrategy::Never
    }

    fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }

    fn loader_mode() -> LoaderMode {
        LoaderMode::Relaxed
    }

    fn load_specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig::default()
    }

    fn partition_buffering_strategy() -> PartitionBuffering {
        PartitionBuffering::Double
    }

    /// Make a convolution config from a convolution problem, and launch options
    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, MatmulSetupError>
    {
        Self::GlobalConvolution::setup::<R, MP>(client, problem, selection, line_sizes)
    }

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        Self::GlobalConvolution::filter_line_sizes(Self::StageMatmul::filter_line_sizes(
            Self::TileMatmul::filter_line_sizes(available_line_sizes),
        ))
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
    ) -> MatmulSelection;
}
