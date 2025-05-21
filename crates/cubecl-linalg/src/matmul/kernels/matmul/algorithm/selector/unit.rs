use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::matmul::components::{MatmulKind, MatmulProblem, MatmulSize, tile::TileMatmulFamily};

use super::MatmulSelection;

const NUM_PLANES_APPROX: u32 = 2;
const ARBITRARY_K_COUNT: u32 = 8;
const TILE_DIM: u32 = 4;

#[derive(Debug)]
pub struct UnitMatmulSelection {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
    pub plane_dim: u32,
}

impl MatmulSelection for UnitMatmulSelection {
    fn tile_shape(&self) -> MatmulSize {
        self.tile_shape
    }

    fn tile_count(&self) -> MatmulSize {
        self.tile_count
    }
}

pub fn unit_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    _client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    _elem_stage: Elem,
    _elem_acc: Elem,
) -> UnitMatmulSelection {
    match Into::<MatmulKind>::into(problem) {
        MatmulKind::General => general_unit_selector::<R>(problem, plane_dim),
        MatmulKind::MatVec => matvec_unit_selector::<R>(problem, plane_dim),
        MatmulKind::VecMat => vecmat_unit_selector::<R>(problem, plane_dim),
        MatmulKind::ScalarVec => scalarvec_unit_selector::<R>(problem, plane_dim),
        MatmulKind::VecScalar => vecscalar_unit_selector::<R>(problem, plane_dim),
        MatmulKind::InnerProduct => inner_product_unit_selector::<R>(problem, plane_dim),
        MatmulKind::OuterProduct => outer_product_unit_selector::<R>(problem, plane_dim),
        MatmulKind::ScalarProduct => scalar_product_unit_selector::<R>(problem, plane_dim),
    }
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
fn general_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (stage_m, stage_n) = closest_factor_pair(num_units);

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: TILE_DIM,
            n: TILE_DIM,
            k: TILE_DIM,
        },
        tile_count: MatmulSize {
            m: stage_m,
            n: stage_n,
            k: ARBITRARY_K_COUNT,
        },
        plane_dim,
    }
}

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: TILE_DIM,
            n: 1,
            k: TILE_DIM,
        },
        tile_count: MatmulSize {
            m: num_units,
            n: 1,
            k: ARBITRARY_K_COUNT,
        },
        plane_dim,
    }
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: 1,
            n: TILE_DIM,
            k: TILE_DIM,
        },
        tile_count: MatmulSize {
            m: 1,
            n: num_units,
            k: ARBITRARY_K_COUNT,
        },
        plane_dim,
    }
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: 1,
            n: TILE_DIM,
            k: 1,
        },
        tile_count: MatmulSize {
            m: 1,
            n: num_units,
            k: 1,
        },
        plane_dim,
    }
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: TILE_DIM,
            n: 1,
            k: 1,
        },
        tile_count: MatmulSize {
            m: num_units,
            n: 1,
            k: 1,
        },
        plane_dim,
    }
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: 1,
            n: 1,
            k: TILE_DIM,
        },
        tile_count: MatmulSize {
            m: 1,
            n: 1,
            k: ARBITRARY_K_COUNT,
        },
        plane_dim,
    }
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (stage_m, stage_n) = closest_factor_pair(num_units);

    UnitMatmulSelection {
        tile_shape: MatmulSize {
            m: TILE_DIM,
            n: TILE_DIM,
            k: 1,
        },
        tile_count: MatmulSize {
            m: stage_m,
            n: stage_n,
            k: 1,
        },
        plane_dim,
    }
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector<R: Runtime>(
    _problem: &MatmulProblem,
    plane_dim: u32,
) -> UnitMatmulSelection {
    UnitMatmulSelection {
        tile_shape: MatmulSize { m: 1, n: 1, k: 1 },
        tile_count: MatmulSize { m: 1, n: 1, k: 1 },
        plane_dim,
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
