use cubecl_core::Feature;
use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};
use cubecl_runtime::DeviceProperties;

use crate::components::batch::{
    CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, SmAllocation,
};
use crate::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};
use crate::components::stage::PartitionBuffering;
use crate::components::{MatmulProblem, tile::TileMatmulFamily};
use crate::components::{
    MatmulSelection, MultiRowStrategy, PartitionSize, StageSize, TileSize, TilingScheme,
};

pub const NUM_SM_APPROX: u32 = 50;
pub const NUM_TENSOR_CORES_APPROX: u32 = 4;

#[derive(Default, Debug)]
/// Options to select the best plane matmul [selection](MatmulSelection).
pub struct PlaneMatmulSelectionOptions {
    pub partition_k: Option<u32>,
    pub specialized: bool,
    pub row_count: Option<u32>,
    pub multi_row_strategy: MultiRowStrategy,
    pub partition_buffering: Option<PartitionBuffering>,
}

pub fn plane_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    elem_stage: Elem,
    elem_acc: Elem,
    options: PlaneMatmulSelectionOptions,
) -> MatmulSelection {
    let tile_size = find_instruction_size(
        if TMM::requires_accelerator() {
            Some((client.properties(), (elem_stage, elem_stage, elem_acc)))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let row_count = options.row_count.unwrap_or_else(|| {
        let max_plane_per_cube = client.properties().hardware.max_units_per_cube / plane_dim;
        let precision_factor = match elem_stage.size() >= 4 {
            true => 2,
            false => 1,
        };
        max_plane_per_cube / (4 * precision_factor)
    });

    let (rows_per_plane, stage_size_m, partition_shape_n) = select_size(
        options.multi_row_strategy,
        row_count as usize,
        tile_size.m() as usize,
        problem.m,
    );

    let tiles_per_partition = PartitionSize::new(
        rows_per_plane as u32,
        partition_shape_n as u32,
        options
            .partition_k
            .unwrap_or_else(|| plane_dim / tile_size.k()),
    );

    let partitions_per_stage = StageSize::new(stage_size_m as u32, 1, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(tiles_per_partition)
        .with_stage_size(partitions_per_stage)
        .build()
        .unwrap();

    let partition_buffering = options.partition_buffering.unwrap_or_else(|| {
        if tiling_scheme.tiles_in_stage_partition_n() > 1 {
            PartitionBuffering::Double
        } else {
            PartitionBuffering::Single
        }
    });

    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanSelection::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanSelection::FromProblem,
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    let mut builder = MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(partition_buffering)
        .hypercube_config(hypercube);

    if options.specialized {
        builder = builder.load_specialization_config(LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        });
    }

    builder.build()
}

fn select_size(
    strategy: MultiRowStrategy,
    plane_count: usize,
    instruction_m: usize,
    problem_m: usize,
) -> (usize, usize, usize) {
    let rows = match strategy {
        MultiRowStrategy::Never => 1,
        MultiRowStrategy::Always(count) => count,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => {
            if problem_m > plane_count * instruction_m * minimum_stage_count as usize {
                2
            } else {
                1
            }
        }
    } as usize;

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
