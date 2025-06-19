use crate::components::batch::BatchMatmulFamily;
use crate::components::global::GlobalMatmulFamily;
use crate::components::stage::StageMatmulFamily;
use crate::components::tile::TileMatmulFamily;
use crate::components::{AvailableLineSizes, MatmulLineSizes, MatmulPrecision, MatmulProblem};
use crate::kernels::MatmulSetupError;
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

use super::MatmulSelection;

#[derive(Debug, Clone)]
pub enum MultiRowStrategy {
    /// Always one row per plane
    Never,
    /// Always multiple rows per plane
    Always,
    /// Uses multiple rows if the `m` dimension of the matmul implies at least the minimum number of stages along `m`
    Adaptive { minimum_stage_count: usize },
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadingPrecomputeStrategy {
    /// Don't precompute anything in loading jobs
    #[default]
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
    type SelectionArgs: Default + Clone;
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalMatmul: GlobalMatmulFamily;
    type BatchMatmul: BatchMatmulFamily;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<<Self::BatchMatmul as BatchMatmulFamily>::Config, MatmulSetupError> {
        Self::BatchMatmul::setup::<MP, R>(client, problem, selection, line_sizes)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        Self::BatchMatmul::filter_line_sizes(Self::GlobalMatmul::filter_line_sizes(
            Self::StageMatmul::filter_line_sizes(Self::TileMatmul::filter_line_sizes(
                available_line_sizes,
            )),
        ))
    }
}
