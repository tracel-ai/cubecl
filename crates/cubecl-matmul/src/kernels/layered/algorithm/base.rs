use crate::components::batch::BatchMatmulFamily;
use crate::components::global::GlobalMatmulFamily;
use crate::components::stage::StageMatmulFamily;
use crate::components::tile::TileMatmulFamily;
use crate::components::{
    AvailableLineSizes, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSelection,
    MatmulSetupError,
};
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

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

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_max
    }
}
