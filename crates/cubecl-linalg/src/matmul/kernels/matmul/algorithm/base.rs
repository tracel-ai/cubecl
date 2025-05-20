use crate::matmul::components::stage::{StageBuffering, StageVectorization};
use crate::matmul::components::{
    CompleteStageTiling, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    batch, global, stage, tile,
};
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

use super::MatmulSelection;

type GlobalInput = (StageInput, LoadingPrecomputeStrategy);

type StageInput = (
    CompleteStageTiling,
    stage::StageBuffering,
    StageVectorization,
    (u32, u32),
);

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
    type TileMatmul: tile::TileMatmulFamily;
    type StageMatmul: stage::StageMatmulFamily<Input = StageInput>;
    type GlobalMatmul: global::GlobalMatmulFamily<Input = GlobalInput>;
    type BatchMatmul: batch::BatchMatmulFamily<Input = GlobalInput>;
    type MatmulSelection: MatmulSelection;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim;
    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount;

    fn line_sizes<R: Runtime>(
        problem: MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
    ) -> MatmulLineSizes {
        MatmulLineSizes::maximize::<R>(problem, in_available, out_available)
    }

    fn num_stages() -> (u32, u32) {
        (1, 1)
    }

    fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }

    fn stage_buffering_strategy() -> StageBuffering {
        StageBuffering::Double
    }

    #[allow(clippy::type_complexity, clippy::result_large_err)]
    fn make_config(
        input: <Self::BatchMatmul as MatmulConfigFactory>::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Result<<Self::BatchMatmul as MatmulConfigFactory>::Config, MatmulLaunchError> {
        let config = Self::BatchMatmul::make_config(
            input, problem, line_sizes, cube_dim, cube_count, quantized,
        );
        problem.check_config(&config)?;
        problem.check_line_sizes(line_sizes)?;
        Self::BatchMatmul::check_config(&config)?;
        Ok(config)
    }

    #[allow(clippy::result_large_err)]
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::BatchMatmul as MatmulConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::BatchMatmul::check_availability::<R, MP>(client, config)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> Self::MatmulSelection;
}
