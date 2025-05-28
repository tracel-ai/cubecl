use crate::matmul::components::MatmulSize;
use crate::matmul::components::stage::{PartitionsPerStage, StageVectorization, TilesPerPartition};
use crate::matmul::components::{MatmulProblem, MatrixLayout};
use crate::matmul::kernels::matmul::{Algorithm, GlobalInput, StageInput, UnitMatmulSelection};
use crate::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;
use crate::matmul::tests::test_utils::TestPrecision;
use cubecl_core::Runtime;

pub fn test_algo<
    A: Algorithm<MatmulSelection = UnitMatmulSelection>,
    P: TestPrecision,
    R: Runtime,
>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_shape: MatmulSize,
    tiles_per_partition: TilesPerPartition,
    partitions_per_stage: PartitionsPerStage,
    stage_k: u32,
    problem: MatmulSize,
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
        m: problem.m as usize,
        n: problem.n as usize,
        k: problem.k as usize,
        batches: (vec![2], vec![2]),
        lhs_layout: layouts.0,
        rhs_layout: layouts.1,
    };

    let selection = UnitMatmulSelection {
        tile_shape,
        tiles_per_partition,
        partitions_per_stage,
        stage_k,
        plane_dim,
    };

    let tiling = (&selection).into();
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };

    test_matmul_algorithm::<A, P, R>(
        client,
        problem,
        GlobalInput {
            stage_input: StageInput {
                tiling_scheme: tiling,
                stage_buffering: A::stage_buffering_strategy(),
                stage_vectorization: vectorization,
                num_stages: A::num_stages(),
                tiles_per_partition,
            },
            loading_precompute_strategy: A::loading_precompute_strategy(),
            loader_mode: A::loader_mode(),
        },
        selection,
    );
}
