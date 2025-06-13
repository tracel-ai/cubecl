use cubecl_matmul::{
    components::{
        AvailableLineSizes, InputIdent, InvalidConfigError, LoadSpecializationConfig,
        MatmulLineSizes, MatmulPrecision,
        global::{args::MatmulArgs, load::LoaderMode},
        stage::{NumStages, PartitionBuffering, StageMatmulFamily, StageVectorization},
        tile::TileMatmulFamily,
    },
    kernels::{
        MatmulAvailabilityError, MatmulSetupError,
        matmul::{
            GlobalInput, LoadingPrecomputeStrategy, MatmulSelection, MultiRowStrategy, StageInput,
        },
    },
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
    type StageMatmul: StageMatmulFamily<Input = StageInput>;
    type GlobalConvolution: ConvolutionFamily<Input = GlobalInput<StageInput>>;

    type Args: MatmulArgs;

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_in_stage_m();
        let n_stage = selection.tiling_scheme.elements_in_stage_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn global_input(selection: &MatmulSelection) -> GlobalInput<StageInput> {
        let partition_buffering = if selection.tiling_scheme.tiles_in_stage_partition_n() > 1 {
            Self::partition_buffering_strategy()
        } else {
            PartitionBuffering::Single
        };

        let stage_vectorization = StageVectorization {
            stage_line_size: 0,
            stage_elem_padding: 0,
        };

        GlobalInput {
            stage_input: StageInput {
                tiling_scheme: selection.tiling_scheme,
                partition_buffering,
                stage_vectorization,
                num_stages: Self::num_stages(),
                load_specialization: Self::load_specialization(),
            },
            loading_precompute_strategy: Self::loading_precompute_strategy(),
            loader_mode: Self::loader_mode(),
        }
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
        LoadSpecializationConfig::None
    }

    fn partition_buffering_strategy() -> PartitionBuffering {
        PartitionBuffering::Double
    }

    /// Make a convolution config from a convolution problem, and launch options
    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, MatmulSetupError>
    {
        Self::GlobalConvolution::setup::<R, MP>(client, problem, selection, available_line_sizes)
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
    ) -> MatmulSelection;

    // fn line_sizes(
    //     problem: &ConvolutionProblem,
    //     in_available: impl Iterator<Item = u8> + Clone,
    //     out_available: impl Iterator<Item = u8> + Clone,
    // ) -> MatmulLineSizes {
    //     MatmulLineSizes::new_maximized(&problem.as_matmul_problem(), in_available, out_available)
    // }
}
