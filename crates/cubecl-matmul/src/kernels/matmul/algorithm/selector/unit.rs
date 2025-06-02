use crate::components::{MatmulKind, MatmulProblem, TilingScheme};

use super::MatmulSelection;

const NUM_PLANES_APPROX: u32 = 2;
const TILE_DIM: u32 = 4;
const PARTITION_K_APPROX: u32 = 4;

#[derive(Debug, Clone)]
pub struct UnitMatmulSelection {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
}

impl MatmulSelection for UnitMatmulSelection {
    fn tiling_scheme(&self) -> &TilingScheme {
        &self.tiling_scheme
    }
}

pub fn unit_matmul_selection(problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    match Into::<MatmulKind>::into(problem) {
        MatmulKind::General => general_unit_selector(problem, plane_dim),
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
fn general_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (stage_size_m, stage_size_n) = closest_factor_pair(num_units);
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, TILE_DIM, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, 1, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((num_units, 1, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, TILE_DIM, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, num_units, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, TILE_DIM, 1).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, num_units, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, 1, 1).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((num_units, 1, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, 1, TILE_DIM).into())
        .with_partition_size((1, 1, PARTITION_K_APPROX).into())
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (stage_size_m, stage_size_n) = closest_factor_pair(num_units);
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((TILE_DIM, TILE_DIM, 1).into())
        .with_partition_size((1, 1, 1).into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((1, 1, 1).into())
        .with_partition_size((1, 1, 1).into())
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();

    UnitMatmulSelection {
        plane_dim,
        tiling_scheme,
    }
}

/// Returns the factor pair `(a, b)` of `n` minimizing their difference,
/// with `a >= b` and `a * b == n`.
fn closest_factor_pair(n: u32) -> (u32, u32) {
    let sqrt_n = (n as f64).sqrt() as u32;
    for a in (1..=sqrt_n).rev() {
        if n % a == 0 {
            return (n / a, a);
        }
    }
    (n, 1)
}
