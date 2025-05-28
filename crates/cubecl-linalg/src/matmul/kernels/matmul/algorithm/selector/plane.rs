use std::cmp::min;

use cubecl_core::Feature;
use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};
use cubecl_runtime::DeviceProperties;

use crate::matmul::components::{MatmulProblem, tile::TileMatmulFamily};
use crate::matmul::components::{PartitionsPerStage, TileShape, TilesPerPartition, TilingScheme};
use crate::matmul::kernels::matmul::MultiRowStrategy;

use super::MatmulSelection;

pub(crate) const NUM_SM_APPROX: u32 = 50;
pub(crate) const NUM_TENSOR_CORES_APPROX: u32 = 4;
const NUM_PLANES_PER_TENSOR_CORES: u32 = 2;

#[derive(Debug)]
pub struct PlaneMatmulSelection {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
}

impl MatmulSelection for PlaneMatmulSelection {
    fn tiling_scheme(&self) -> TilingScheme {
        self.tiling_scheme.clone()
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
    let tile_shape = find_instruction_shape(
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

    let partition_shape_n = find_tiles_in_partition_n(
        problem.m,
        problem.n,
        problem.num_batches(),
        client
            .properties()
            .hardware
            .num_streaming_multiprocessors
            .unwrap_or(NUM_SM_APPROX) as usize,
        tensor_cores_channels,
        tile_shape.m as usize,
        tile_shape.n as usize,
    );

    let (rows_per_plane, stage_size_m) = change_rows_per_plane(
        multi_row_strategy,
        partition_shape_n,
        tile_shape.m as usize,
        problem.m,
    );

    let tiles_per_partition = TilesPerPartition {
        m: rows_per_plane as u32,
        n: partition_shape_n as u32,
    };

    let partitions_per_stage = PartitionsPerStage {
        m: stage_size_m as u32,
        n: 1,
    };

    // Makes all rows the length of plane_dim
    let stage_k = plane_dim / tile_shape.k;

    let tiling_scheme = TilingScheme::builder()
        .with_tile_shape(tile_shape)
        .with_tiles_per_partition(tiles_per_partition)
        .with_partitions_per_stage(partitions_per_stage)
        .with_stage_k_tile_count(stage_k)
        .build()
        .unwrap();

    PlaneMatmulSelection {
        tiling_scheme,
        plane_dim,
    }
}

fn change_rows_per_plane(
    strategy: MultiRowStrategy,
    total_stage_size_wanted: usize,
    instruction_m: usize,
    problem_m: usize,
) -> (usize, usize) {
    let use_multi = match strategy {
        MultiRowStrategy::Never => false,
        MultiRowStrategy::Always => true,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => problem_m > total_stage_size_wanted * instruction_m * minimum_stage_count,
    };

    let rows = if use_multi { 2 } else { 1 };
    (rows, total_stage_size_wanted * rows)
}

/// A heuristic to find the number of tiles in the stage.
///
/// Maximizes tensor core usage unless doing so would significantly impair
/// parallelization across SMs. It ensures the number of cubes is as close as
/// possible to the available SMs.
pub(crate) fn find_tiles_in_partition_n(
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
) -> TileShape {
    let supported = |m: u8, n: u8, k: u8| {
        properties
            .map(|(p, (a, b, c))| p.feature_enabled(Feature::Cmma { a, b, c, m, n, k }))
            .unwrap_or(true)
    };

    if m >= 4 * n && supported(32, 8, 16) {
        (32, 8, 16).into()
    } else if n >= 4 * n && supported(8, 32, 16) {
        (8, 32, 16).into()
    } else if supported(16, 16, 16) {
        (16, 16, 16).into()
    } else if supported(8, 8, 8) {
        (8, 8, 8).into()
    } else {
        (16, 16, 8).into()
    }
}
