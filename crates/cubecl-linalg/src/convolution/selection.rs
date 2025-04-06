use cubecl_core::{Runtime, client::ComputeClient};

use super::{
    algorithm::{Algorithm, StageInput},
    base::ConvolutionProblem,
};
use crate::matmul::{
    components::{CompleteStageTiling, MatmulPrecision, MatmulSelection, stage},
    kernels::matmul::matmul_selection,
};

pub fn select_matmul<A: Algorithm, R: Runtime, MP: MatmulPrecision>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &ConvolutionProblem,
    plane_dim: u32,
) -> (MatmulSelection, StageInput) {
    let mm_problem = problem.as_matmul_problem();
    let selection = matmul_selection::<A::TileMatmul, MP, R>(client, &mm_problem, plane_dim);
    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape,
        tile_count: selection.tile_count,
    };

    // TODO Allows to select double buffering
    (selection, (config_input, stage::Buffering::Single))
}
