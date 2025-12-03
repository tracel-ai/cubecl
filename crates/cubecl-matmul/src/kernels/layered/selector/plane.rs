use cubecl_core::{Runtime, client::ComputeClient, ir::StorageType};
use cubecl_runtime::MmaConfig;

use crate::components::{
    MatmulAvailabilityError, MatmulElems, MatmulSelection, MatmulSetupError, MultiRowStrategy,
    PartitionSize, StageSize, TileSize, TilingScheme, adjust_dtypes,
};
use crate::components::{MatmulLineSizes, stage::PartitionBuffering};
use crate::components::{MatmulProblem, tile::TileMatmulFamily};
use crate::components::{
    MatrixLayout,
    batch::{CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, SmAllocation},
    stage::SwizzleMode,
};
use crate::components::{
    SwizzleConfig,
    global::{LoadSpecializationConfig, SpecializationTensorConfig},
};
use crate::kernels::layered::selector::is_tiny;

pub const NUM_SM_APPROX: u32 = 50;
pub const NUM_TENSOR_CORES_APPROX: u32 = 4;

#[derive(Default, Debug)]
/// Options to select the best plane matmul [selection](MatmulSelection).
pub struct PlaneMatmulSelectionOptions {
    pub partition_k: Option<u32>,
    pub specialized: bool,
    pub swizzled: bool,
    pub row_count: Option<u32>,
    pub multi_row_strategy: MultiRowStrategy,
    pub partition_buffering: Option<PartitionBuffering>,
    /// Enables the tiny selector when the [matmul problem](MatmulProblem) is flagged as tiny.
    pub tiny_selection_enabled: bool,
}

pub fn plane_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    dtypes: &mut MatmulElems,
    line_sizes: &MatmulLineSizes,
    options: PlaneMatmulSelectionOptions,
) -> Result<MatmulSelection, MatmulSetupError> {
    adjust_dtypes(client, dtypes, TMM::requires_accelerator());

    if plane_dim == 1 {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim: 1 },
        ));
    }

    let tile_size = find_instruction_size::<R, TMM>(client, dtypes, problem.m, problem.n)?;

    if options.tiny_selection_enabled && is_tiny(problem, &tile_size) {
        return Ok(selection_tiny(client, problem, tile_size, plane_dim));
    }

    let row_count = options.row_count.unwrap_or_else(|| {
        let max_plane_per_cube = client.properties().hardware.max_units_per_cube / plane_dim;
        // Compensate for register use
        let precision_factor = match dtypes.lhs_stage.size() >= 4 {
            true => 2,
            false => 1,
        };
        let mut tile_factor = tile_size.n().div_ceil(4);
        if problem.m as u32 <= tile_size.m() * 4 || problem.n as u32 <= tile_size.n() * 4 {
            tile_factor = 8;
        }
        max_plane_per_cube / (tile_factor * precision_factor)
    });

    if row_count == 0 {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim },
        ));
    }

    let (rows_per_plane, mut stage_size_m, mut partition_shape_n) = select_size(
        options.multi_row_strategy,
        row_count as usize,
        tile_size.m() as usize,
        problem.m,
    );

    if options.swizzled {
        if problem.lhs_layout == MatrixLayout::ColMajor {
            let elem_size = dtypes.lhs_global.size();
            while partition_shape_n * tile_size.n() as usize * elem_size > 128 {
                partition_shape_n /= 2;
                stage_size_m /= 2;
            }
        }
        if problem.rhs_layout == MatrixLayout::RowMajor {
            let elem_size = dtypes.rhs_global.size();
            while partition_shape_n * tile_size.n() as usize * elem_size > 128 {
                partition_shape_n /= 2;
                stage_size_m /= 2;
            }
        }
    }

    let mut partition_shape_k = options
        .partition_k
        .unwrap_or_else(|| plane_dim / tile_size.k());

    if options.swizzled {
        if problem.lhs_layout == MatrixLayout::RowMajor {
            let elem_size = dtypes.lhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > 128 {
                partition_shape_k /= 2;
            }
        }
        if problem.rhs_layout == MatrixLayout::ColMajor {
            let elem_size = dtypes.rhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > 128 {
                partition_shape_k /= 2;
            }
        }
    }

    let tiles_per_partition = PartitionSize::new(
        rows_per_plane as u32,
        partition_shape_n as u32,
        partition_shape_k,
    );

    let partitions_per_stage = StageSize::new(stage_size_m as u32, 1, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(tiles_per_partition)
        .with_stage_size(partitions_per_stage)
        .build()
        .unwrap();

    let partition_buffering = options.partition_buffering.unwrap_or_else(|| {
        if tiling_scheme.tiles_per_stage_partition_along_n() > 1 {
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

    if options.swizzled {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m(),
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k(),
        };

        let lhs = select_swizzle(lhs_swizzle_dim, *dtypes.lhs_stage, line_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, *dtypes.rhs_stage, line_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleConfig {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok(builder.build())
}

/// All modes currently use atom size 16
const SWIZZLE_ATOM: usize = 16;

pub fn select_swizzle(swizzle_dim: u32, elem: StorageType, line_size: u8) -> SwizzleMode {
    // Line size exceeds swizzle atom
    if elem.size() * line_size as usize > SWIZZLE_ATOM {
        return SwizzleMode::None;
    }
    let swizzle_dim_bytes = swizzle_dim as usize * elem.size();
    if !swizzle_dim_bytes.is_power_of_two() {
        return SwizzleMode::None;
    }
    match swizzle_dim_bytes {
        32 => SwizzleMode::B32,
        64 => SwizzleMode::B64,
        128 => SwizzleMode::B128,
        _ => SwizzleMode::None,
    }
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
pub fn find_instruction_size<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    elems: &MatmulElems,
    m: usize,
    n: usize,
) -> Result<TileSize, MatmulAvailabilityError> {
    let supported = |m: u32, n: u32, k: u32| {
        TMM::is_supported(
            client,
            MmaConfig {
                a_type: *elems.lhs_register,
                b_type: *elems.rhs_register,
                cd_type: *elems.acc_register,
                m,
                n,
                k,
            },
        )
    };

    let val = if m >= 4 * n && supported(32, 8, 16) {
        (32, 8, 16).into()
    } else if n >= 4 * n && supported(8, 32, 16) {
        (8, 32, 16).into()
    } else if supported(16, 16, 16) {
        (16, 16, 16).into()
    } else if supported(8, 8, 8) {
        (8, 8, 8).into()
    } else {
        match TMM::supported_sizes(
            client,
            *elems.lhs_register,
            *elems.rhs_register,
            *elems.acc_register,
        )
        .first()
        .copied()
        {
            Some(val) => val,
            None => return Err(MatmulAvailabilityError::TileSizeNotFound),
        }
    };

    Ok(val)
}

fn selection_tiny<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> MatmulSelection {
    // If the K axis is big, we can leverage that.
    let pk = u32::min(problem.k as u32 / tile_size.k(), 8);
    let pk = u32::max(pk, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, pk))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
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
            w: 2,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .build()
}
