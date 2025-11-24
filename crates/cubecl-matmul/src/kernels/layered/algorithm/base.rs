use crate::components::batch::BatchMatmulFamily;
use crate::components::global::GlobalMatmulFamily;
use crate::components::stage::StageMatmulFamily;
use crate::components::tile::TileMatmulFamily;
use crate::components::{
    AvailableLineSizes, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection,
    MatmulSetupError,
};
use cubecl_core::prelude::*;

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type SelectionArgs: Default + Clone;
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalMatmul: GlobalMatmulFamily;
    type BatchMatmul: BatchMatmulFamily;

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<<Self::BatchMatmul as BatchMatmulFamily>::Config, MatmulSetupError> {
        Self::BatchMatmul::setup(client, problem, selection, line_sizes, dtypes)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::SelectionArgs,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        Self::BatchMatmul::filter_line_sizes(Self::GlobalMatmul::filter_line_sizes(
            Self::StageMatmul::filter_line_sizes(Self::TileMatmul::filter_line_sizes(
                available_line_sizes,
            )),
        ))
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_max
    }
}
