use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionProblem, AttentionSelection, AttentionTilingScheme, batch::HypercubeSelection,
    },
    kernels::Algorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub struct TestOptions {
    pub reuse_key_value: bool,
    pub two_rows_in_array_tile: bool,
}

impl Default for TestOptions {
    fn default() -> Self {
        Self {
            reuse_key_value: false,
            two_rows_in_array_tile: false,
        }
    }
}

pub fn attention_test_launch<A: Algorithm, R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    tiling_scheme: AttentionTilingScheme,
    problem: AttentionProblem,
    test_options: TestOptions,
) {
    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme,
        reuse_key_value: test_options.reuse_key_value,
        two_rows_in_array_tile: test_options.two_rows_in_array_tile,
    };

    test_attention_algorithm::<A, (f32, f32), R>(client, problem, selection);
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
            use cubecl_attention::kernels::dummy::{
                DummyAcceleratedAlgorithm, DummyRegisterAlgorithm,
            };
            use $crate::tests::macros::{TestOptions, attention_test_launch};

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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                )
            }

            #[cfg(target_os = "macos")]
            #[test]
            #[ignore = "accelerated disabled"]
            fn attention_8_8_8_8_accelerated() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyAcceleratedAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                )
            }

            #[test]
            fn attention_two_rows_in_array_tile() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    TestOptions {
                        two_rows_in_array_tile: true,
                        ..Default::default()
                    },
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
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_partition_vd2() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[cfg(target_os = "macos")]
            #[test]
            #[ignore = "Accelerated disabled"]
            fn attention_partition2_global2_accelerated() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize * 2,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * 2,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyAcceleratedAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                )
            }

            #[cfg(target_os = "macos")]
            #[test]
            #[ignore = "Accelerated disabled"]
            fn attention_partition_q2_stage2_accelerated() {
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
                let stage_size = AttentionStageSize { seq_q: 2 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyAcceleratedAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                )
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * num_iterations,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * num_iterations,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_partition_kv1_global1_with_oob() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize - 1,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
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
                let stage_size = AttentionStageSize { seq_q: 2 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * 2 + 1,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_partition_oob_in_q() {
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
                    seq_q: 1,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    // seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize + 9,
                    seq_kv: 8,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_stage4() {
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
                let stage_size = AttentionStageSize { seq_q: 4 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_stage2_problem4() {
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize * 2,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * 2,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
            fn attention_stage2_partition_all2() {
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
                let stage_size = AttentionStageSize { seq_q: 2 };
                let tiling_scheme = AttentionTilingScheme {
                    tile_size,
                    partition_size,
                    stage_size,
                };
                let problem = AttentionProblem {
                    batch: 1,
                    num_heads: 1,
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    Default::default(),
                );
            }

            #[test]
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
                    seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                    seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
                    head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                    val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                    masked: false,
                };
                attention_test_launch::<DummyRegisterAlgorithm, TestRuntime>(
                    client,
                    tiling_scheme,
                    problem,
                    TestOptions {
                        reuse_key_value: true,
                        ..Default::default()
                    },
                );
            }

            // #[test]
            // fn attention_double_row_wise() {
            //     let client = TestRuntime::client(&Default::default());
            //     let tile_size = AttentionTileSize {
            //         seq_q: 16,
            //         seq_kv: 16,
            //         head_dim: 16,
            //         val_dim: 16,
            //     };
            //     let partition_size = AttentionPartitionSize {
            //         seq_q: 2,
            //         seq_kv: 2,
            //         head_dim: 2,
            //         val_dim: 2,
            //     };
            //     let stage_size = AttentionStageSize { seq_q: 2 };
            //     let tiling_scheme = AttentionTilingScheme {
            //         tile_size,
            //         partition_size,
            //         stage_size,
            //     };
            //     let problem = AttentionProblem {
            //         batch: 1,
            //         num_heads: 1,
            //         seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
            //         seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize,
            //         head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
            //         val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
            //         masked: false,
            //     };
            //     attention_test_launch::<DummyDoubleRegisterAlgorithm, TestRuntime>(
            //         client,
            //         tiling_scheme,
            //         problem,
            //         true,
            //     );
            // }
        }
    };
}
