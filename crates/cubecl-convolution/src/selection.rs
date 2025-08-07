use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::stage::PartitionBuffering;

use super::base::ConvolutionProblem;
use cubecl_matmul::components::{MatmulElems, MatmulSelection, TilingScheme};
use cubecl_matmul::{
    components::tile::TileMatmulFamily,
    kernels::layered::{NUM_SM_APPROX, NUM_TENSOR_CORES_APPROX, find_instruction_size},
};

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

pub fn convolution_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &ConvolutionProblem,
    plane_dim: u32,
    matmul_elems: MatmulElems,
) -> MatmulSelection {
    // rough heuristic based on previous bench results where 512 channels with a 3x3 kernel seemed
    // to be the rough cutoff for the k=4 size.
    let stage_k = if problem.k >= 4096 { 4 } else { 2 };

    let tile_size = find_instruction_size(
        if TMM::requires_accelerator() {
            Some((
                client.properties(),
                (
                    matmul_elems.lhs_register,
                    matmul_elems.rhs_register,
                    matmul_elems.acc,
                ),
            ))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let hardware = &client.properties().hardware;
    let num_sm = hardware
        .num_streaming_multiprocessors
        .unwrap_or(NUM_TENSOR_CORES_APPROX);
    let max_tensor_cores = hardware.num_tensor_cores.unwrap_or(NUM_SM_APPROX);

    let (stage_size_m, stage_size_n) = find_stage_size_m_n(
        problem.m,
        problem.n,
        num_sm as usize,
        max_tensor_cores as usize,
        tile_size.m() as usize,
        tile_size.n() as usize,
        stage_k as usize,
    );

    let tiling_scheme = TilingScheme::builder()
        .with_stage_size((stage_size_m as u32, 1, 1).into())
        .with_tile_size(tile_size)
        .with_partition_size((1, stage_size_n as u32, stage_k).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .build()
}
