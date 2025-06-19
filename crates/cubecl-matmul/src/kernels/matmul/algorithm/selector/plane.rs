use cubecl_core::Feature;
use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};
use cubecl_runtime::DeviceProperties;

use crate::components::stage::PartitionBuffering;
use crate::components::{MatmulProblem, tile::TileMatmulFamily};
use crate::components::{PartitionSize, StageSize, TileSize, TilingScheme};
use crate::kernels::matmul::MultiRowStrategy;

use super::MatmulSelection;

pub const NUM_SM_APPROX: u32 = 50;
pub const NUM_TENSOR_CORES_APPROX: u32 = 4;

pub fn plane_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    multi_row_strategy: MultiRowStrategy,
    elem_stage: Elem,
    elem_acc: Elem,
) -> MatmulSelection {
    let tile_size = find_instruction_size(
        if TMM::requires_tensor_cores() {
            Some((client.properties(), (elem_stage, elem_stage, elem_acc)))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let occupancy_factor = 4;
    let max_units_per_cube = client.properties().hardware.max_units_per_cube;
    let plane_count = max_units_per_cube / (plane_dim * occupancy_factor);

    let (rows_per_plane, stage_size_m, partition_shape_n) = select_size(
        multi_row_strategy,
        plane_count as usize,
        tile_size.m() as usize,
        problem.m,
    );

    let tiles_per_partition = PartitionSize::new(
        rows_per_plane as u32,
        partition_shape_n as u32,
        plane_dim / tile_size.k(),
    );

    let partitions_per_stage = StageSize::new(stage_size_m as u32, 1, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(tiles_per_partition)
        .with_stage_size(partitions_per_stage)
        .build()
        .unwrap();

    let partition_buffering = if tiling_scheme.tiles_in_stage_partition_n() > 1 {
        PartitionBuffering::Double
    } else {
        PartitionBuffering::Single
    };

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(partition_buffering)
        .build()
}

fn select_size(
    strategy: MultiRowStrategy,
    plane_count: usize,
    instruction_m: usize,
    problem_m: usize,
) -> (usize, usize, usize) {
    let use_multi = match strategy {
        MultiRowStrategy::Never => false,
        MultiRowStrategy::Always => true,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => problem_m > plane_count * instruction_m * minimum_stage_count,
    };

    let rows = if use_multi { 2 } else { 1 };
    (rows, plane_count / rows, plane_count)
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
#[allow(clippy::type_complexity)]
pub fn find_instruction_size(
    properties: Option<(&DeviceProperties<Feature>, (Elem, Elem, Elem))>,
    m: usize,
    n: usize,
) -> TileSize {
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
