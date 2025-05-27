use crate::matmul::components::{
    MatmulKind, MatmulProblem, MatmulSize,
    stage::{PartitionsPerStage, TilesPerPartition},
};

use super::MatmulSelection;

const NUM_PLANES_APPROX: u32 = 2;
const ARBITRARY_K_COUNT: u32 = 8;
const TILE_DIM: u32 = 4;
const TILES_PER_PARTITION_APPROX: TilesPerPartition = TilesPerPartition { m: 1, n: 1 };

#[derive(Debug)]
pub struct UnitMatmulSelection {
    pub plane_dim: u32,
    pub tile_shape: MatmulSize,
    pub tiles_per_partition: TilesPerPartition,
    pub partitions_per_stage: PartitionsPerStage,
    pub stage_k: u32,
}

impl MatmulSelection for UnitMatmulSelection {
    fn tile_shape(&self) -> MatmulSize {
        self.tile_shape
    }

    fn tile_count(&self) -> MatmulSize {
        MatmulSize {
            m: self.tiles_per_partition.m * self.partitions_per_stage.m,
            n: self.tiles_per_partition.n * self.partitions_per_stage.n,
            k: self.stage_k,
        }
    }

    fn tiles_per_partition(&self) -> TilesPerPartition {
        self.tiles_per_partition
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
    let tile_shape = MatmulSize {
        m: TILE_DIM,
        n: TILE_DIM,
        k: TILE_DIM,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (partition_m, partition_n) = closest_factor_pair(num_units);
    let partitions_per_stage = PartitionsPerStage {
        m: partition_m,
        n: partition_n,
    };
    let stage_k = ARBITRARY_K_COUNT;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

// TODO all other variants

/// (M, K) @ (K, 1) → (M, 1)
fn matvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: TILE_DIM,
        n: 1,
        k: TILE_DIM,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let partitions_per_stage = PartitionsPerStage { m: num_units, n: 1 };
    let stage_k = ARBITRARY_K_COUNT;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (1, K) @ (K, N) → (1, N)
fn vecmat_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: 1,
        n: TILE_DIM,
        k: TILE_DIM,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let partitions_per_stage = PartitionsPerStage { m: 1, n: num_units };
    let stage_k = ARBITRARY_K_COUNT;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (1, 1) @ (1, N) → (1, N)
fn scalarvec_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: 1,
        n: TILE_DIM,
        k: 1,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let partitions_per_stage = PartitionsPerStage { m: 1, n: num_units };
    let stage_k = 1;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (M, 1) @ (1, 1) → (M, 1)
fn vecscalar_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: TILE_DIM,
        n: 1,
        k: 1,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let partitions_per_stage = PartitionsPerStage { m: num_units, n: 1 };
    let stage_k = 1;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (1, K) @ (K, 1) → (1, 1)
fn inner_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: 1,
        n: 1,
        k: TILE_DIM,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let partitions_per_stage = PartitionsPerStage { m: 1, n: 1 };
    let stage_k = ARBITRARY_K_COUNT;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (M, 1) @ (1, N) → (M, N)
fn outer_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize {
        m: TILE_DIM,
        n: TILE_DIM,
        k: 1,
    };
    let tiles_per_partition = TILES_PER_PARTITION_APPROX;

    let num_units = NUM_PLANES_APPROX * plane_dim;
    let (partition_m, partition_n) = closest_factor_pair(num_units);
    let partitions_per_stage = PartitionsPerStage {
        m: partition_m,
        n: partition_n,
    };
    let stage_k = 1;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
    }
}

/// (1, 1) @ (1, 1) → (1, 1)
fn scalar_product_unit_selector(_problem: &MatmulProblem, plane_dim: u32) -> UnitMatmulSelection {
    let tile_shape = MatmulSize { m: 1, n: 1, k: 1 };
    let tiles_per_partition = TilesPerPartition { m: 1, n: 1 };

    let partitions_per_stage = PartitionsPerStage { m: 1, n: 1 };
    let stage_k = 1;

    UnitMatmulSelection {
        plane_dim,
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
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
