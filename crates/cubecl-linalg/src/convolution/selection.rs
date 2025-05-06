use cubecl_core::{Runtime, client::ComputeClient, prelude::*};

use super::{
    algorithm::{Algorithm, StageInput},
    base::ConvolutionProblem,
};
use crate::matmul::{
    components::{
        CompleteStageTiling, MatmulPrecision, MatmulProblem, MatmulSelection, MatmulSize,
        stage::StageVectorization, tile::TileMatmulFamily,
    },
    kernels::matmul::{NUM_SM_APPROX, NUM_TENSOR_CORES_APPROX, find_instruction_shape},
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
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    (
        selection,
        (
            config_input,
            A::stage_buffering_strategy(),
            vectorization,
            A::num_stages(),
        ),
    )
}

/// A heuristic to find the number of tiles in the stage.
///
/// Maximizes tensor core usage unless doing so would significantly impair
/// parallelization across SMs. It ensures the number of cubes is as close as
/// possible to the available SMs.
pub(crate) fn find_stage_size_m_n(
    m: usize,
    n: usize,
    num_sm: usize,
    max_tensor_cores: usize,
    instruction_m: usize,
    instruction_n: usize,
    stage_size_k: usize,
) -> (usize, usize) {
    let max_tiles_elems_m = 256 / instruction_m;
    let max_tiles_elems_n = 256 / instruction_n;
    let max_tiles_total_stage = 16 / stage_size_k;

    let mut dim_num_tiles_m = max_tensor_cores
        .min(max_tiles_elems_m)
        .min(max_tiles_total_stage);

    let mut dim_num_tiles_n = max_tensor_cores
        .min(max_tiles_elems_n)
        .min(max_tiles_total_stage);

    let total_tiles_m = m.div_ceil(instruction_m);
    let total_tiles_n = n.div_ceil(instruction_n);

    while total_tiles_n < dim_num_tiles_n && dim_num_tiles_n > 1 {
        dim_num_tiles_n /= 2;
    }

    let total_tiles = total_tiles_m * total_tiles_n;

    let mut stage_num_tiles = dim_num_tiles_m * dim_num_tiles_n;
    let mut num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;

    // We keep track of two configurations to select the closest to `num_sm`, whether it's a bit over or under
    let mut previous_dim_num_tiles = dim_num_tiles_m;
    let mut previous_num_cubes = num_cubes_expected;

    // Refine tensor core usage to stay as close as possible to `num_sm`
    while num_cubes_expected < num_sm && dim_num_tiles_m > 1 {
        previous_dim_num_tiles = dim_num_tiles_m;
        previous_num_cubes = num_cubes_expected;

        // Reduce tensor core usage
        dim_num_tiles_m = (dim_num_tiles_m + 1) / 2;
        stage_num_tiles = dim_num_tiles_m * dim_num_tiles_n;

        // Number of cubes grows as a consequence of smaller stage
        num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;
    }

    // Compare previous and current values to determine the closest to `num_sm`
    if (previous_num_cubes as isize - num_sm as isize).abs()
        <= (num_cubes_expected as isize - num_sm as isize).abs()
    {
        (previous_dim_num_tiles, dim_num_tiles_n)
    } else {
        (dim_num_tiles_n, dim_num_tiles_m)
    }
}

pub fn matmul_selection<TMM: TileMatmulFamily, MP: MatmulPrecision, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
) -> MatmulSelection {
    // rough heuristic based on previous bench results where 512 channels with a 3x3 kernel seemed
    // to be the rough cutoff for the k=4 size.
    let stage_size_k = if problem.k >= 4096 { 4 } else { 2 };

    let (instruction_m, instruction_n, _instruction_k) = find_instruction_shape(
        if TMM::requires_tensor_cores() {
            Some((
                client.properties(),
                (
                    MP::ES::as_elem_native_unchecked(),
                    MP::ES::as_elem_native_unchecked(),
                    MP::EA::as_elem_native_unchecked(),
                ),
            ))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let hardware = client.properties().hardware_properties();
    let num_sm = hardware
        .num_streaming_multiprocessors
        .unwrap_or(NUM_TENSOR_CORES_APPROX);
    let max_tensor_cores = hardware.num_tensor_cores.unwrap_or(NUM_SM_APPROX);

    let (stage_size_m, stage_size_n) = find_stage_size_m_n(
        problem.m,
        problem.n,
        num_sm as usize,
        max_tensor_cores as usize,
        instruction_m,
        instruction_n,
        stage_size_k,
    );

    let tile_shape = MatmulSize {
        m: stage_size_m as u32,
        n: stage_size_n as u32,
        k: stage_size_k as u32,
    };

    let tile_count = MatmulSize {
        m: (problem.m as u32 + stage_size_m as u32 - 1) / stage_size_m as u32,
        n: (problem.n as u32 + stage_size_n as u32 - 1) / stage_size_n as u32,
        k: (problem.k as u32 + stage_size_k as u32 - 1) / stage_size_k as u32,
    };

    MatmulSelection {
        tile_shape,
        tile_count,
        plane_dim,
        rows_per_plane: 1,
    }
}
