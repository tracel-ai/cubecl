use std::fmt::Display;

use crate::components::{
    MatmulElems, MatmulKind, MatmulLineSizes, MatmulProblem, MatmulSelection, MatrixLayout,
    SwizzleConfig, TilingScheme,
    batch::{CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, SmAllocation},
    stage::{PartitionBuffering, SwizzleMode},
};
use cubecl_core::{Runtime, client::ComputeClient, ir::StorageType};

#[derive(Default, Clone, Copy, Debug)]
pub enum TileSizeSelection {
    // Chooses the smallest tile size possible.
    MinTileSize,
    #[default]
    // Chooses the biggest tile size possible.
    MaxTileSize,
}

impl Display for TileSizeSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileSizeSelection::MinTileSize => f.write_str("min_tile_size"),
            TileSizeSelection::MaxTileSize => f.write_str("max_tile_size"),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub enum PartitionScaling {
    #[default]
    Enabled,
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub enum StageScaling {
    Enabled(u8),
    #[default]
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct UnitMatmulSelectionOptions {
    pub tile: TileSizeSelection,
    pub stage: StageScaling,
    pub partition: PartitionScaling,
    pub swizzle: bool,
}

/// Computes a [MatmulSelection] depending on the problem kind
pub fn unit_matmul_selection<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    line_sizes: &MatmulLineSizes,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
) -> MatmulSelection {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;
    let min_tile_size = u8::max(line_sizes.lhs, line_sizes.rhs);
    let min_tile_size = u8::max(line_sizes.out, min_tile_size) as u32;
    let tile_size = u32::max(min_tile_size, 4);

    match kind {
        MatmulKind::General => general_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::MatVec => matvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::VecMat => vecmat_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::ScalarVec => scalarvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::VecScalar => vecscalar_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::InnerProduct => inner_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::OuterProduct => outer_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
        MatmulKind::ScalarProduct => scalar_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
            dtypes,
            line_sizes,
        ),
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
#[allow(clippy::too_many_arguments)]
fn general_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    use MatrixLayout::*;

    // Manually tested for good performance on many shapes.
    let (tile_size, mut partition_size) =
        match (problem.lhs_layout, problem.rhs_layout, options.tile) {
            (RowMajor, _, TileSizeSelection::MinTileSize) => (
                (1, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 4, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 10),
                ),
            ),
            (ColMajor, RowMajor, TileSizeSelection::MinTileSize) => (
                (tile_size, tile_size, 1),
                (2, 2, scale_partition(options.partition, problem.k, 3, 10)),
            ),
            (ColMajor, ColMajor, _) | (_, _, TileSizeSelection::MaxTileSize) => (
                (tile_size, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 2, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 9),
                ),
            ),
        };

    let mut num_plane = 8;

    if double_buffering {
        if partition_size.0 > 2 {
            partition_size.0 /= 2;
        }
        if partition_size.2 > 2 {
            partition_size.2 /= 2;
        }
        num_plane /= 2;
    }

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane,
        },
        num_sms,
        GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        },
        options.stage,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (M, K) @ (K, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (MatrixLayout::RowMajor, _) => ((1, 1, tile_size), (1, 1, tile_size * 2)),
        _ => ((tile_size, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: plane_dim / 2,
            n: 2,
        },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (1, K) @ (K, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn vecmat_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((1, tile_size, tile_size), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: plane_dim / 2,
        },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (1, 1) @ (1, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn scalarvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (RowMajor, ColMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, ColMajor) => ((1, tile_size, tile_size), (2, 2, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: plane_dim / 2,
        },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn vecscalar_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((tile_size, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: plane_dim / 2,
            n: 2,
        },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (1, K) @ (K, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn inner_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (RowMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: plane_dim, n: 1 }, // TODO: most planes does nothing.
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (M, 1) @ (1, N) → (M, N)
#[allow(clippy::too_many_arguments)]
fn outer_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((tile_size, tile_size, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn scalar_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    _tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((1, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane: 1,
        },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        line_sizes,
    )
}

enum StageSelection {
    WithPlane { plane_dim: u32, num_plane: u32 },
    Fixed { m: u32, n: u32 },
}

impl StageSelection {
    fn into_stages(self) -> (u32, u32) {
        match self {
            StageSelection::WithPlane {
                plane_dim: plane_size,
                num_plane: num_planes,
            } => {
                let num_units = num_planes * plane_size;
                closest_factor_pair(num_units)
            }
            StageSelection::Fixed { m, n } => (m, n),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn selection(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    buffering: PartitionBuffering,
    plane_dim: u32,
    stage: StageSelection,
    num_sms: Option<u32>,
    global_order_config: GlobalOrderSelection,
    stage_scaling: StageScaling,
    swizzle: bool,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> MatmulSelection {
    let (stage_size_m, stage_size_n) = stage.into_stages();

    let (stage_size_m, stage_size_n) = match stage_scaling {
        StageScaling::Enabled(f) => (stage_size_m / f as u32, stage_size_n / f as u32),
        StageScaling::Disabled => (stage_size_m, stage_size_n),
    };

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(t.into())
        .with_partition_size(p.into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    let cube_count_plan = match num_sms {
        Some(num_sms) => CubeCountPlanSelection::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: false,
        },
        None => CubeCountPlanSelection::Flattened,
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(global_order_config)
        .cube_count_plan(cube_count_plan)
        .build();

    let mut builder = MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(buffering)
        .hypercube_config(hypercube);

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m(),
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k(),
        };

        builder = builder.shared_swizzle(SwizzleConfig {
            lhs: select_swizzle(lhs_swizzle_dim, *dtypes.lhs_stage, line_sizes.lhs),
            rhs: select_swizzle(rhs_swizzle_dim, *dtypes.rhs_stage, line_sizes.rhs),
            ..Default::default()
        })
    }

    builder.build()
}

/// All modes currently use atom size 16
const SWIZZLE_ATOM: usize = 16;

fn select_swizzle(swizzle_dim: u32, elem: StorageType, line_size: u8) -> SwizzleMode {
    // Can't swizzle if line size > swizzle atom
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
        _ => SwizzleMode::B128,
    }
}

/// Returns the factor pair `(a, b)` of `n` minimizing their difference,
/// with `a >= b` and `a * b == n`.
pub fn closest_factor_pair(n: u32) -> (u32, u32) {
    let sqrt_n = (n as f64).sqrt() as u32;
    for a in (1..=sqrt_n).rev() {
        if n.is_multiple_of(a) {
            return (n / a, a);
        }
    }
    (n, 1)
}

fn scale_partition(setting: PartitionScaling, axis: usize, max_exp: u32, div_exp: u32) -> u32 {
    if let PartitionScaling::Disabled = setting {
        return 2u32.pow(max_exp);
    }

    let exp = u32::min((axis as u32 / 2u32.pow(div_exp)) + 1, max_exp);
    2u32.pow(exp)
}
