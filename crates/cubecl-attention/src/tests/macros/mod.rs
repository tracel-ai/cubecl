use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionProblem, AttentionSelection, AttentionTilingScheme, batch::HypercubeSelection,
    },
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_test_launch<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    tiling_scheme: AttentionTilingScheme,
    problem: AttentionProblem,
) {
    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme,
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

#[macro_export]
macro_rules! testgen_attention {
    () => {
        #[cfg(feature = "attention_tests")]
        mod attention {
            use super::*;
            use cubecl_attention::components::{
                AttentionPartitionSize, AttentionProblem, AttentionStageSize, AttentionTileSize,
                AttentionTilingScheme,
            };

            #[test]
            fn attention_8_8_8_8() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_9_9_9_9() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 9,
                    seq_kv: 9,
                    head_dim: 9,
                    val_dim: 9,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_7_3_10_10() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 7,
                    seq_kv: 3,
                    head_dim: 10,
                    val_dim: 10,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_8_kv64() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: 64,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_8_kv58() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: 58,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_8_kv5() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: 5,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_8_q16() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: 16,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_8_q4() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: 4,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_partition_q2() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 2,
                    seq_kv: 1,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize,
                    head_dim: tiling_scheme.head_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                )
            }

            #[test]
            fn attention_partition_hd2() {}

            #[test]
            fn attention_partition_kv2() {}
        }
    };
}
