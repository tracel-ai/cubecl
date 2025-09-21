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
    reuse_key_value: bool,
) {
    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme,
        reuse_key_value,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                )
            }

            #[test]
            fn attention_partition_hd2() {
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
                    head_dim: 2,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                )
            }

            #[test]
            fn attention_partition_kv2() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 3,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            fn attention_partition_vd2() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 5,
                    head_dim: 7,
                    val_dim: 4,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 1,
                    head_dim: 7,
                    val_dim: 8,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            fn attention_partition_all2() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 2,
                    seq_kv: 2,
                    head_dim: 2,
                    val_dim: 2,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            fn attention_global_2() {
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
                let num_iterations = 2;
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize * num_iterations,
                    head_dim: tiling_scheme.head_dim() as usize,
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            fn attention_partition_kv2_global_2() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 2,
                    head_dim: 1,
                    val_dim: 1,
                };
                let stage_size = AttentionStageSize { seq_q: 1 };
                let num_iterations = 2;
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.seq_q() as usize,
                    seq_kv: tiling_scheme.seq_kv() as usize * num_iterations,
                    head_dim: tiling_scheme.head_dim() as usize,
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            fn attention_partition_kv1_global2_with_oob() {
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
                    seq_kv: tiling_scheme.seq_kv() as usize * 2 + 1,
                    head_dim: tiling_scheme.head_dim() as usize,
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            #[ignore = "TODO"]
            fn attention_partition_kv2_with_oob() {
                let client = TestRuntime::client(&Default::default());
                let tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                let partition_size = AttentionPartitionSize {
                    seq_q: 1,
                    seq_kv: 2,
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
                    seq_kv: tiling_scheme.seq_kv() as usize + 1,
                    head_dim: tiling_scheme.head_dim() as usize,
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            #[ignore = "TODO"]
            fn attention_stage2() {
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
                let stage_size = AttentionStageSize { seq_q: 2 };
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    false,
                );
            }

            #[test]
            #[ignore = "TODO"]
            fn attention_reuse_key_value() {
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
                    head_dim: 2,
                    val_dim: 2,
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
                    val_dim: tiling_scheme.val_dim() as usize,
                    masked: false,
                };
                $crate::tests::macros::attention_test_launch::<TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    true,
                );
            }
        }
    };
}
