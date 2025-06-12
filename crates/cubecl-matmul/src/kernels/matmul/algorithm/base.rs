use crate::components::batch::BatchMatmulFamily;
use crate::components::global::GlobalMatmulFamily;
use crate::components::global::load::LoaderMode;
use crate::components::stage::{
    NumStages, PartitionBuffering, StageMatmulFamily, StageVectorization,
};
use crate::components::tile::TileMatmulFamily;
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    TilingScheme, stage,
};
use crate::components::{LoadSpecializationConfig, MatmulChecker};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

use super::MatmulSelection;

pub struct GlobalInput<SI> {
    pub stage_input: SI,
    // Decided by selector
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    // Decided by selector
    pub loader_mode: LoaderMode,
}

pub struct StageInput {
    // Decided by selector
    pub tiling_scheme: TilingScheme,
    // Decided by selector
    pub partition_buffering: stage::PartitionBuffering,
    // Decided by selector
    pub stage_vectorization: StageVectorization,
    // Decided by global
    pub num_stages: NumStages,
    // Decided by selector
    pub load_specialization: LoadSpecializationConfig,
}

pub enum MultiRowStrategy {
    /// Always one row per plane
    Never,
    /// Always multiple rows per plane
    Always,
    /// Uses multiple rows if the `m` dimension of the matmul implies at least the minimum number of stages along `m`
    Adaptive { minimum_stage_count: usize },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadingPrecomputeStrategy {
    /// Don't precompute anything in loading jobs
    Never,
    /// Precompute values that are shared across tasks
    Always,
}

impl Default for LoadingPrecomputeStrategy {
    fn default() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }
}

impl From<LoadingPrecomputeStrategy> for bool {
    fn from(strategy: LoadingPrecomputeStrategy) -> Self {
        match strategy {
            LoadingPrecomputeStrategy::Always => true,
            LoadingPrecomputeStrategy::Never => false,
        }
    }
}

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily<Input = StageInput>;
    type GlobalMatmul: GlobalMatmulFamily<Input = GlobalInput<StageInput>>;
    type BatchMatmul: BatchMatmulFamily<Input = GlobalInput<StageInput>>;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
    ) -> Result<<Self::BatchMatmul as MatmulChecker>::Config, MatmulSetupError> {
        Self::BatchMatmul::setup(problem, selection, available_line_sizes)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        Self::TileMatmul::selection::<R>(client, problem, plane_dim, elem_stage, elem_acc)
    }

    // fn cube_dim(selection: &MatmulSelection) -> Result<CubeDim, InvalidConfigError> {
    //     Self::GlobalMatmul::cube_dim(selection, Self::load_specialization_config())
    // }

    // fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
    //     Self::BatchMatmul::cube_count(selection, problem)
    // }

    // fn line_sizes(
    //     problem: &MatmulProblem,
    //     in_available: impl Iterator<Item = u8> + Clone,
    //     out_available: impl Iterator<Item = u8> + Clone,
    //     _selection: &MatmulSelection,
    // ) -> MatmulLineSizes {
    //     MatmulLineSizes::new_maximized(problem, in_available, out_available)
    // }

    // fn global_input(selection: &MatmulSelection) -> GlobalInput<StageInput> {
    //     let partition_buffering = if selection.tiling_scheme.tiles_in_stage_partition_n() > 1 {
    //         Self::partition_buffering_strategy()
    //     } else {
    //         PartitionBuffering::Single
    //     };

    //     let stage_vectorization = StageVectorization {
    //         stage_line_size: 0,
    //         stage_elem_padding: 0,
    //     };

    //     GlobalInput {
    //         stage_input: StageInput {
    //             tiling_scheme: selection.tiling_scheme,
    //             partition_buffering,
    //             stage_vectorization,
    //             num_stages: Self::num_stages(),
    //             load_specialization: Self::load_specialization_config(),
    //         },
    //         loading_precompute_strategy: Self::loading_precompute_strategy(),
    //         loader_mode: Self::loader_mode(),
    //     }
    // }

    // fn num_stages() -> NumStages {
    //     (1, 1).into()
    // }

    // fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
    //     LoadingPrecomputeStrategy::Never
    // }

    // fn loader_mode() -> LoaderMode {
    //     LoaderMode::Relaxed
    // }

    // fn load_specialization_config() -> LoadSpecializationConfig {
    //     LoadSpecializationConfig::None
    // }

    // fn partition_buffering_strategy() -> PartitionBuffering {
    //     PartitionBuffering::Double
    // }

    // #[allow(clippy::type_complexity, clippy::result_large_err)]
    // fn setup(
    //     // input: <Self::BatchMatmul as BatchMatmulFamily>::Input,
    //     problem: &MatmulProblem,
    //     // Decided at tile
    //     // But max line sizes come from problem
    // line_sizes: &MatmulLineSizes,
    //     // Decided at global (use proxy cubedim)
    //     cube_dim: &CubeDim,
    //     // Decided at batch (use proxy cubecount)
    //     cube_count: &CubeCount,
    //     // Decided at selection
    //     quantized: bool,
    // ) -> Result<<Self::BatchMatmul as MatmulChecker>::Config, MatmulLaunchError> {
    //     #[cfg(target_os = "macos")]
    //     if cube_dim.num_elems() >= 512 {
    //         return Err(MatmulLaunchError::Unavailable(
    //             MatmulAvailabilityError::CubeDimTooBig(*cube_dim),
    //         ));
    //     }

    //     let config =
    //         Self::BatchMatmul::setup(input, problem, line_sizes, cube_dim, cube_count, quantized);
    //     Self::BatchMatmul::check_config(&config)?;
    //     problem.check_config(&config)?;
    //     problem.check_line_sizes(line_sizes)?;
    //     Ok(config)
    // }

    // #[allow(clippy::result_large_err)]
    // fn check_availability<R: Runtime, MP: MatmulPrecision>(
    //     client: &ComputeClient<R::Server, R::Channel>,
    //     config: &<Self::BatchMatmul as MatmulChecker>::Config,
    // ) -> Result<(), MatmulAvailabilityError> {
    //     Self::BatchMatmul::check_availability::<R, MP>(client, config)
    // }

    // fn selection<R: Runtime>(
    //     client: &ComputeClient<R::Server, R::Channel>,
    //     problem: &MatmulProblem,
    //     plane_dim: u32,
    //     elem_stage: Elem,
    //     elem_acc: Elem,
    // ) -> MatmulSelection;
}
