use crate::components::{
    MatmulKind, MatmulLineSizes, MatmulProblem, MatmulSelection, MatrixLayout, TilingScheme,
    batch::{CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, SmAllocation},
    stage::PartitionBuffering,
};
use cubecl_core::{Runtime, client::ComputeClient};

#[derive(Default, Clone, Copy, Debug)]
pub enum TileSizeSelection {
    // Chooses the smallest tile size possible.
    MinTileSize,
    #[default]
    // Chooses the biggest tile size possible.
    MaxTileSize,
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
}

/// Computes a [MatmulSelection] depending on the problem kind
pub fn unit_matmul_selection<R: Runtime>(
    client: &ComputeClient<R::Server>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    line_size: &MatmulLineSizes,
    options: UnitMatmulSelectionOptions,
) -> MatmulSelection {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;
    let min_tile_size = u8::max(line_size.lhs, line_size.rhs);
    let min_tile_size = u8::max(line_size.out, min_tile_size) as u32;
    let tile_size = u32::max(min_tile_size, 4);

    match kind {
        MatmulKind::General => general_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
        ),
        MatmulKind::MatVec => matvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
        ),
        MatmulKind::VecMat => vecmat_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            options,
        ),
        MatmulKind::ScalarVec => {
            scalarvec_unit_selector(problem, plane_dim, double_buffering, tile_size, num_sms)
        }
        MatmulKind::VecScalar => {
            vecscalar_unit_selector(problem, plane_dim, double_buffering, tile_size, num_sms)
        }
        MatmulKind::InnerProduct => {
            inner_product_unit_selector(problem, plane_dim, double_buffering, tile_size, num_sms)
        }
        MatmulKind::OuterProduct => {
            outer_product_unit_selector(problem, plane_dim, double_buffering, tile_size, num_sms)
        }
        MatmulKind::ScalarProduct => {
            scalar_product_unit_selector(problem, plane_dim, double_buffering, tile_size, num_sms)
        }
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
fn general_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    options: UnitMatmulSelectionOptions,
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
    )
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    _options: UnitMatmulSelectionOptions,
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
    )
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    _options: UnitMatmulSelectionOptions,
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
    )
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
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
        StageSelection::Fixed { m: 1, n: plane_dim },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((tile_size, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: plane_dim, n: 1 },
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
    )
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
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
        StageSelection::Fixed { m: plane_dim, n: 1 }, // TODO: BAD
        num_sms,
        GlobalOrderSelection::Default,
        StageScaling::Disabled,
    )
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
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
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    _tile_size: u32,
    num_sms: Option<u32>,
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
            sm_usage: SmAllocation::Full,
            cubes_first: false,
        },
        None => CubeCountPlanSelection::Flattened,
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(global_order_config)
        .cube_count_plan(cube_count_plan)
        .build();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(buffering)
        .hypercube_config(hypercube)
        .build()
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
