use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::{
    MatmulKind, MatmulProblem, MatrixLayout, TilingScheme,
    batch::{CubeCountPlanConfig, GlobalOrderConfig, HypercubeConfig, SmAllocation},
    stage::PartitionBuffering,
};

use super::MatmulSelection;

pub fn unit_matmul_selection<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
) -> MatmulSelection {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;

    match kind {
        MatmulKind::General => general_unit_selector(problem, plane_dim, double_buffering, num_sms),
        MatmulKind::MatVec => matvec_unit_selector(problem, plane_dim, double_buffering, num_sms),
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
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, mut partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 4, 4), (16, 2, 4)),
        (RowMajor, ColMajor) => ((1, 4, 4), (16, 2, 4)),
        (ColMajor, RowMajor) => ((4, 4, 1), (2, 2, 8)),
        (ColMajor, ColMajor) => ((4, 4, 4), (4, 2, 4)),
    };

    // It seems to be faster, it's not a requirement of the algo.
    if double_buffering {
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
        problem,
    )
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    num_sms: Option<u32>,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 1, 4), (1, 1, 4)),
        (RowMajor, ColMajor) => ((1, 1, 4), (1, 1, 4)),
        (ColMajor, RowMajor) => ((4, 1, 4), (1, 1, 4)),
        (ColMajor, ColMajor) => ((4, 1, 4), (1, 1, 4)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        problem,
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
        (RowMajor, ColMajor) => ((1, 4, 4), (2, 1, 8)),
        (ColMajor, RowMajor) => ((1, 4, 4), (1, 1, 4)),
        (ColMajor, ColMajor) => ((1, 4, 4), (2, 1, 8)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        problem,
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
        problem,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(
    problem: &MatmulProblem,
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
        problem,
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
        problem,
    )
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(
    problem: &MatmulProblem,
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
        problem,
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(
    problem: &MatmulProblem,
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
        problem,
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
    problem: &MatmulProblem,
) -> MatmulSelection {
    let (stage_size_m, stage_size_n) = stage.into_stages();

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(t.into())
        .with_partition_size(p.into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    let cube_count_plan = match num_sms {
        Some(num_sms) => CubeCountPlanConfig::CubeFirst {
            num_sms,
            sm_usage: SmAllocation::Exact,
        },
        None => CubeCountPlanConfig::Flattened,
    };

    let hypercube = HypercubeConfig::builder(&tiling_scheme)
        .global_order(GlobalOrderConfig::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
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
