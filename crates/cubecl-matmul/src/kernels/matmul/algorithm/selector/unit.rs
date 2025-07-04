use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::{
    MatmulKind, MatmulProblem, MatrixLayout, TilingScheme,
    batch::{CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, SmAllocation},
    stage::PartitionBuffering,
};

use super::MatmulSelection;

#[derive(Default, Clone, Copy, Debug)]
pub enum TileSizeSelection {
    // Choses the smallest tile size possible.
    MinTileSize,
    #[default]
    // Choses the biggest tile size possible.
    MaxTileSize,
}

pub fn unit_matmul_selection<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: TileSizeSelection,
) -> MatmulSelection {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;

    match kind {
        MatmulKind::General => {
            general_unit_selector(problem, plane_dim, double_buffering, num_sms, tile_size)
        }
        MatmulKind::MatVec => {
            matvec_unit_selector(problem, plane_dim, double_buffering, num_sms, tile_size)
        }
        MatmulKind::VecMat => vecmat_unit_selector(problem, plane_dim, double_buffering, num_sms),
        MatmulKind::ScalarVec => {
            scalarvec_unit_selector(problem, plane_dim, double_buffering, num_sms)
        }
        MatmulKind::VecScalar => {
            vecscalar_unit_selector(problem, plane_dim, double_buffering, num_sms)
        }
        MatmulKind::InnerProduct => {
            inner_product_unit_selector(problem, plane_dim, double_buffering, num_sms)
        }
        MatmulKind::OuterProduct => {
            outer_product_unit_selector(problem, plane_dim, double_buffering, num_sms)
        }
        MatmulKind::ScalarProduct => {
            scalar_product_unit_selector(problem, plane_dim, double_buffering, num_sms)
        }
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
fn general_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    num_sms: Option<u32>,
    tile_size_selection: TileSizeSelection,
) -> MatmulSelection {
    use MatrixLayout::*;

    // Manually tested for good performance on many shapes.
    let (tile_size, mut partition_size) =
        match (problem.lhs_layout, problem.rhs_layout, tile_size_selection) {
            (RowMajor, _, TileSizeSelection::MinTileSize) => (
                (1, 4, 4),
                (
                    scale_partition(problem.m, 4, 9),
                    2,
                    scale_partition(problem.k, 2, 10),
                ),
            ),
            (ColMajor, RowMajor, TileSizeSelection::MinTileSize) => {
                ((4, 4, 1), (2, 2, scale_partition(problem.k, 3, 10)))
            }
            (ColMajor, ColMajor, _) | (_, _, TileSizeSelection::MaxTileSize) => (
                (4, 4, 4),
                (
                    scale_partition(problem.m, 2, 9),
                    2,
                    scale_partition(problem.k, 2, 9),
                ),
            ),
        };

    // It seems to be faster, it's not a requirement of the algo.
    if double_buffering && partition_size.2 > 2 {
        partition_size.2 /= 2;
    }

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane: 8,
        },
        num_sms,
        GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        },
    )
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
    tile_size_selection: TileSizeSelection,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) =
        match (problem.lhs_layout, problem.rhs_layout, tile_size_selection) {
            (RowMajor, _, TileSizeSelection::MinTileSize) => ((1, 1, 4), (1, 1, 4)),
            _ => ((4, 1, 4), (1, 1, 4)),
        };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 4, 4), (1, 1, 4)),
        (RowMajor, ColMajor) => ((1, 4, 4), (2, 1, scale_partition(problem.k, 3, 7))),
        (ColMajor, RowMajor) => ((1, 4, 4), (1, 1, 4)),
        (ColMajor, ColMajor) => ((1, 4, 4), (2, 1, scale_partition(problem.k, 3, 7))),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 4, 4), (1, 2, 1)),
        (RowMajor, ColMajor) => ((1, 4, 4), (1, 2, 1)),
        (ColMajor, RowMajor) => ((1, 4, 4), (1, 2, 1)),
        (ColMajor, ColMajor) => ((1, 4, 4), (2, 2, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 4, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((4, 1, 4), (1, 1, 2));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 4 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 1, 4), (1, 1, 1)),
        (RowMajor, ColMajor) => ((1, 1, 4), (1, 1, 1)),
        (ColMajor, RowMajor) => ((1, 1, 4), (1, 1, 1)),
        (ColMajor, ColMajor) => ((1, 1, 4), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 4, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((4, 4, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        GlobalOrderSelection::Default,
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(
    _problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
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
            num_plane: 8,
        },
        num_sms,
        GlobalOrderSelection::Default,
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

fn selection(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    buffering: PartitionBuffering,
    plane_dim: u32,
    stage: StageSelection,
    num_sms: Option<u32>,
    global_order_config: GlobalOrderSelection,
) -> MatmulSelection {
    let (stage_size_m, stage_size_n) = stage.into_stages();

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
            cubes_first: true,
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
        if n % a == 0 {
            return (n / a, a);
        }
    }
    (n, 1)
}

fn scale_partition(axis: usize, max_exp: u32, div_exp: u32) -> u32 {
    let exp = u32::min((axis as u32 / 2u32.pow(div_exp)) + 1, max_exp);
    2u32.pow(exp)
}
