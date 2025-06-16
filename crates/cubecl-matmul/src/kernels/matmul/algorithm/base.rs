use crate::components::batch::BatchMatmulFamily;
use crate::components::global::GlobalMatmulFamily;
use crate::components::stage::StageMatmulFamily;
use crate::components::tile::TileMatmulFamily;
use crate::components::{AvailableLineSizes, MatmulPrecision, MatmulProblem};
use crate::kernels::MatmulSetupError;
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

use super::MatmulSelection;

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
    type StageMatmul: StageMatmulFamily;
    type GlobalMatmul: GlobalMatmulFamily;
    type BatchMatmul: BatchMatmulFamily;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<<Self::BatchMatmul as BatchMatmulFamily>::Config, MatmulSetupError> {
        Self::BatchMatmul::setup::<MP, R>(client, problem, selection, available_line_sizes)
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

    fn cube_count(
        problem: &MatmulProblem,
        config: &<Self::BatchMatmul as BatchMatmulFamily>::Config,
    ) -> CubeCount {
        problem.cube_count::<Self::BatchMatmul>(config)
    }
}
