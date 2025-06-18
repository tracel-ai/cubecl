use crate::{
    components::{
        MatmulKind, MatmulLayouts, MatmulProblem, MatrixLayout, TilingScheme,
        stage::PartitionBuffering,
    },
    tune_key::MatmulGlobalScale,
};

use super::MatmulSelection;

const NUM_PLANES_APPROX: u32 = 8;
const TILE_DIM: u32 = 4;
const PARTITION_K_APPROX: u32 = 2;

pub fn unit_matmul_selection(
    problem: &MatmulProblem,
    line_sizes: MatmulLayouts,
    plane_dim: u32,
    double_buffering: bool,
) -> MatmulSelection {
    match Into::<MatmulKind>::into(problem) {
        MatmulKind::General => {
            general_unit_selector(problem, line_sizes, plane_dim, double_buffering)
        }
        MatmulKind::MatVec => matvec_unit_selector(problem, plane_dim),
        MatmulKind::VecMat => vecmat_unit_selector(problem, plane_dim),
        MatmulKind::ScalarVec => scalarvec_unit_selector(problem, plane_dim),
        MatmulKind::VecScalar => vecscalar_unit_selector(problem, plane_dim),
        MatmulKind::InnerProduct => inner_product_unit_selector(problem, plane_dim),
        MatmulKind::OuterProduct => outer_product_unit_selector(problem, plane_dim),
        MatmulKind::ScalarProduct => scalar_product_unit_selector(problem, plane_dim),
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
fn general_unit_selector(
    problem: &MatmulProblem,
    layout: MatmulLayouts,
    plane_dim: u32,
    double_buffering: bool,
) -> MatmulSelection {
    use MatrixLayout::*;
    let (tile_size, mut partition_size) = match (layout.lhs, layout.rhs) {
        (RowMajor, RowMajor) => ((1, 4, 4), (16, 2, 4)),
        (RowMajor, ColMajor) => ((1, 4, 4), (16, 2, 4)),
        (ColMajor, RowMajor) => ((4, 4, 1), (2, 2, 8)),
        (ColMajor, ColMajor) => ((4, 4, 4), (4, 2, 4)),
    };

    // It seems to be faster, it's not a requirement of the algo.
    if double_buffering {
        partition_size.2 /= 2;
    }

    let scale = MatmulGlobalScale::from_size(problem.m, problem.n, problem.k);
    match scale {
        MatmulGlobalScale::Large | MatmulGlobalScale::Medium => selection(
            tile_size,
            partition_size,
            PartitionBuffering::Single,
            plane_dim,
            8,
        ),
        MatmulGlobalScale::Small => selection(
            tile_size,
            partition_size,
            PartitionBuffering::Single,
            plane_dim,
            8,
        ),
    }
}

fn selection(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    buffering: PartitionBuffering,
    plane_dim: u32,
    num_planes: u32,
) -> MatmulSelection {
    let num_units = num_planes * plane_dim;
    let (stage_size_m, stage_size_n) = closest_factor_pair(num_units);

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

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, 1, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((num_units, 1, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, TILE_DIM, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, num_units, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, TILE_DIM, 1).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, num_units, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, 1, 1).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((num_units, 1, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, 1, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (stage_size_m, stage_size_n) = closest_factor_pair(num_units);
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, TILE_DIM, 1).into())
        .with_partition_size((1, 1, 1).into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> MatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, 1, 1).into())
        .with_partition_size((1, 1, 1).into())
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();

    MatmulSelection::builder(tiling_scheme, plane_dim).build()
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
