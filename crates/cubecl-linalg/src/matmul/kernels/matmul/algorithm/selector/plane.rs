use std::cmp::min;

use cubecl_core::Feature;
use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};
use cubecl_runtime::DeviceProperties;

use crate::matmul::components::{MatmulProblem, MatmulSize, tile::TileMatmulFamily};
use crate::matmul::kernels::matmul::MultiRowStrategy;

use super::MatmulSelection;

pub(crate) const NUM_SM_APPROX: u32 = 50;
pub(crate) const NUM_TENSOR_CORES_APPROX: u32 = 4;
const NUM_PLANES_PER_TENSOR_CORES: u32 = 2;

#[derive(Debug)]
pub struct PlaneMatmulSelection {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
    pub plane_dim: u32,
    pub rows_per_plane: u32,
}

impl MatmulSelection for PlaneMatmulSelection {
    fn tile_shape(&self) -> MatmulSize {
        self.tile_shape
    }

    fn tile_count(&self) -> MatmulSize {
        self.tile_count
    }
}

pub fn plane_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    multi_row_strategy: MultiRowStrategy,
    elem_stage: Elem,
    elem_acc: Elem,
) -> PlaneMatmulSelection {
    let (instruction_m, instruction_n, instruction_k) = find_instruction_shape(
        if TMM::requires_tensor_cores() {
            Some((client.properties(), (elem_stage, elem_stage, elem_acc)))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let num_tensor_cores = client
        .properties()
        .hardware
        .num_tensor_cores
        .unwrap_or(NUM_TENSOR_CORES_APPROX);
    // The number of planes that can send tasks to tensor cores.
    //
    // Going over 8 might use too much shared memory.
    let tensor_cores_channels = min(8, num_tensor_cores * NUM_PLANES_PER_TENSOR_CORES) as usize;

    let stage_size = find_stage_size(
        problem.m,
        problem.n,
        problem.num_batches(),
        client
            .properties()
            .hardware
            .num_streaming_multiprocessors
            .unwrap_or(NUM_SM_APPROX) as usize,
        tensor_cores_channels,
        instruction_m,
        instruction_n,
    );

    let (rows_per_plane, stage_size_m) =
        change_rows_per_plane(multi_row_strategy, stage_size, instruction_m, problem.m);

    // Makes all rows the length of plane_dim
    let k = plane_dim / instruction_k as u32;

    PlaneMatmulSelection {
        tile_shape: MatmulSize {
            m: instruction_m as u32,
            n: instruction_n as u32,
            k: instruction_k as u32,
        },
        tile_count: MatmulSize {
            m: stage_size_m as u32,
            n: stage_size as u32,
            k,
        },
        plane_dim,
        rows_per_plane: rows_per_plane as u32,
    }
}

fn change_rows_per_plane(
    strategy: MultiRowStrategy,
    stage_size: usize,
    instruction_m: usize,
    problem_m: usize,
) -> (usize, usize) {
    let use_multi = match strategy {
        MultiRowStrategy::Never => false,
        MultiRowStrategy::Always => true,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => problem_m > stage_size * instruction_m * minimum_stage_count,
    };

    let rows = if use_multi { 2 } else { 1 };
    (rows, stage_size * rows)
}

/// A heuristic to find the number of tiles in the stage.
///
/// Maximizes tensor core usage unless doing so would significantly impair
/// parallelization across SMs. It ensures the number of cubes is as close as
/// possible to the available SMs.
pub(crate) fn find_stage_size(
    m: usize,
    n: usize,
    num_batches: usize,
    num_sm: usize,
    virtual_tensor_cores: usize,
    instruction_m: usize,
    instruction_n: usize,
) -> usize {
    let min_inst = instruction_m.min(instruction_n);
    let max_tiles = 256 / min_inst;
    let mut dim_num_tiles = virtual_tensor_cores.min(max_tiles);

    let total_tiles_m = (m + instruction_m - 1) / instruction_m;
    let total_tiles_n = (n + instruction_n - 1) / instruction_n;

    let total_tiles = total_tiles_m * total_tiles_n * num_batches;

    let mut stage_num_tiles = dim_num_tiles * dim_num_tiles;
    let mut num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;

    // We keep track of two configurations to select the closest to `num_sm`, whether it's a bit over or under
    let mut previous_dim_num_tiles = dim_num_tiles;
    let mut previous_num_cubes = num_cubes_expected;

    // Refine tensor core usage to stay as close as possible to `num_sm`
    while num_cubes_expected < num_sm && stage_num_tiles > 1 {
        previous_dim_num_tiles = dim_num_tiles;
        previous_num_cubes = num_cubes_expected;

        // Reduce tensor core usage
        dim_num_tiles = (dim_num_tiles + 1) / 2;
        stage_num_tiles = dim_num_tiles * dim_num_tiles;

        // Number of cubes grows as a consequence of smaller stage
        num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;
    }

    // Compare previous and current values to determine the closest to `num_sm`
    if (previous_num_cubes as isize - num_sm as isize).abs()
        <= (num_cubes_expected as isize - num_sm as isize).abs()
    {
        previous_dim_num_tiles
    } else {
        dim_num_tiles
    }
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
#[allow(clippy::type_complexity)]
pub(crate) fn find_instruction_shape(
    properties: Option<(&DeviceProperties<Feature>, (Elem, Elem, Elem))>,
    m: usize,
    n: usize,
) -> (usize, usize, usize) {
    match properties {
        Some(properties) => {
            let supported = |m: u8, n: u8, k: u8| {
                let (p, (a, b, c)) = properties;
                p.feature_enabled(Feature::Cmma { a, b, c, m, n, k })
            };

            if m >= 4 * n && supported(32, 8, 16) {
                (32, 8, 16)
            } else if n >= 4 * n && supported(8, 32, 16) {
                (8, 32, 16)
            } else if supported(16, 16, 16) {
                (16, 16, 16)
            } else if supported(8, 8, 8) {
                (8, 8, 8)
            } else {
                (16, 16, 8)
            }
        } // TODO: instead, make another selector for non-cmma
        // -> Better: every Algorithm implements its selector
        // Though most will call the same function
        // In refactoring selector, never be generic over element type, only use DTYPE
        // Also, never use TensorHandleRef
        //
        // For unit selector,
        // We first want to choose the stage size, to have a good unit count
        // Then, we want a small tile size
        // Depends on problem, like if it's mat@vec
        //
        // None => (8, 8, 8),
        None => (4, 4, 4),
    }
}
