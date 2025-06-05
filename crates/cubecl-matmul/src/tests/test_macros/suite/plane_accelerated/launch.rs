use crate::components::{MatmulProblem, MatrixLayout, PartitionSize, StageSize, TileSize};
use crate::components::{MatmulProblemSize, TilingScheme};
use crate::kernels::matmul::{Algorithm, MatmulSelection};
use crate::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;
use crate::tests::test_utils::TestPrecision;
use cubecl_core::Runtime;

pub fn test_algo<A: Algorithm, P: TestPrecision, R: Runtime>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_size: TileSize,
    tiles_per_partition: PartitionSize,
    partitions_per_stage: StageSize,
    problem_size: MatmulProblemSize,
) {
    let client = R::client(&Default::default());
    let plane_dim = match client.properties().hardware.defined_plane_size() {
        Some(val) => val,
        None => {
            println!("Can't run test without a fixed plane size.");
            return;
        }
    };

    let problem = MatmulProblem {
        m: problem_size.m() as usize,
        n: problem_size.n() as usize,
        k: problem_size.k() as usize,
        batches: (vec![2], vec![2]),
        lhs_layout: layouts.0,
        rhs_layout: layouts.1,
    };

    let tiling_scheme = TilingScheme::builder()
        .with_stage_size(partitions_per_stage)
        .with_tile_size(tile_size)
        .with_partition_size(tiles_per_partition)
        .build()
        .unwrap();

    let selection = MatmulSelection {
        tiling_scheme: tiling_scheme.clone(),
        plane_dim,
    };

    test_matmul_algorithm::<A, P, R>(client, problem, A::global_input(&selection), selection);
}
