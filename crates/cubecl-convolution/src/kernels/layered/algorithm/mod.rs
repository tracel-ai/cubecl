use cubecl_matmul::components::{
    AvailableLineSizes, LoadingPrecomputeStrategy, MatmulElems, MatmulIdent, MatmulLineSizes,
    MatmulPrecision, MatmulSelection, MatmulSetupError, MultiRowStrategy,
    global::{LoadSpecializationConfig, args::MatmulArgs, read::ReaderMode},
    stage::{NumStages, PartitionBuffering, StageMatmulFamily},
    tile::TileMatmulFamily,
};

use cubecl_std::tensor::TensorHandle;

use cubecl_core::prelude::*;

use crate::components::{
    ConvolutionProblem,
    global::{GlobalConfig, GlobalConvolutionFamily},
};

pub mod multi_stage_tma;
pub mod simple;
pub mod simple_tma;

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalConvolution: GlobalConvolutionFamily;

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

    fn reader_mode() -> ReaderMode {
        ReaderMode::Relaxed
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
    ) -> Result<GlobalConfig<Self::GlobalConvolution>, MatmulSetupError> {
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
        ident: MatmulIdent,
    ) -> TensorHandle<R, E>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        matmul_elems: MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError>;
}
