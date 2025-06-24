use crate::components::batch::{
    CubeDistributionConfig, GlobalOrder, HypercubeConfig, SmAllocation,
};
use crate::components::stage::PartitionBuffering;
use crate::components::{
    LoadSpecializationConfig, MatmulProblem, MatrixLayout, PartitionSize, StageSize, TileSize,
};
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
    load_specialization_config: LoadSpecializationConfig,
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

    let partition_buffering = if tiling_scheme.tiles_in_stage_partition_n() > 1 {
        PartitionBuffering::Double
    } else {
        PartitionBuffering::Single
    };

    let selection = MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(partition_buffering)
        .load_specialization_config(load_specialization_config)
        .hypercube_config(
            HypercubeConfig::builder(&tiling_scheme)
                .global_order(GlobalOrder::SwizzleColMajor(2))
                .cube_distribution(CubeDistributionConfig::SmFirst {
                    num_sms: 19,
                    sm_usage: SmAllocation::Full,
                })
                .build(),
        )
        .build();

    test_matmul_algorithm::<A, P, R>(client, problem, selection);
}
