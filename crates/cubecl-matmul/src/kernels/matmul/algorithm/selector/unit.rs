use crate::components::{
    MatmulKind, MatmulLayouts, MatmulProblem, MatrixLayout, TilingScheme, stage::PartitionBuffering,
};

use super::MatmulSelection;

pub fn unit_matmul_selection(
    problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    double_buffering: bool,
) -> MatmulSelection {
    let kind: MatmulKind = problem.into();
    match kind {
        MatmulKind::General => general_unit_selector(problem, layouts, plane_dim, double_buffering),
        MatmulKind::MatVec => matvec_unit_selector(problem, layouts, plane_dim, double_buffering),
        MatmulKind::VecMat => vecmat_unit_selector(problem, layouts, plane_dim, double_buffering),
        MatmulKind::ScalarVec => {
            scalarvec_unit_selector(problem, layouts, plane_dim, double_buffering)
        }
        MatmulKind::VecScalar => {
            vecscalar_unit_selector(problem, layouts, plane_dim, double_buffering)
        }
        MatmulKind::InnerProduct => {
            inner_product_unit_selector(problem, layouts, plane_dim, double_buffering)
        }
        MatmulKind::OuterProduct => {
            outer_product_unit_selector(problem, layouts, plane_dim, double_buffering)
        }
        MatmulKind::ScalarProduct => {
            scalar_product_unit_selector(problem, layouts, plane_dim, double_buffering)
        }
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
fn general_unit_selector(
    _problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, mut partition_size) = match (layouts.lhs, layouts.rhs) {
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
    )
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(
    _problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (layouts.lhs, layouts.rhs) {
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
    )
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(
    _problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (layouts.lhs, layouts.rhs) {
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
    )
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(
    _problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (layouts.lhs, layouts.rhs) {
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
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(
    _problem: &MatmulProblem,
    _layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((4, 1, 4), (1, 1, 2));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 4 },
    )
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(
    _problem: &MatmulProblem,
    layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, partition_size) = match (layouts.lhs, layouts.rhs) {
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
    )
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(
    _problem: &MatmulProblem,
    _layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
) -> MatmulSelection {
    let (tile_size, partition_size) = ((4, 4, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(
    _problem: &MatmulProblem,
    _layouts: MatmulLayouts,
    plane_dim: u32,
    _double_buffering: bool,
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
) -> MatmulSelection {
    let (stage_size_m, stage_size_n) = stage.into_stages();

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(t.into())
        .with_partition_size(p.into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(buffering)
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
