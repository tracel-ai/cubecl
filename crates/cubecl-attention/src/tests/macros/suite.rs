#[macro_export]
macro_rules! testgen_attention_suite {
    () => {
        use super::*;
        use cubecl_attention::components::{
            AttentionPartitionSize, AttentionProblem, AttentionStageSize, AttentionTileSize,
            AttentionTilingScheme,
        };
        use $crate::tests::macros::{TestOptions, attention_test_launch};

        #[test]
        fn attention_one_tile() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_two_rows_in_array_tile() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
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
        fn attention_one_tile_seqq16() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_one_tile_seqq4() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_partition_seqq2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 2,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_partition_hd2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 2,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_partition_kv2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 3,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_vd2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 2,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_all2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 2,
                seq_kv: 2,
                head_dim: 2,
                val_dim: 2,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_global_2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let num_iterations = 2;
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv2_global_2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 2,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let num_iterations = 2;
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv1_global1_with_oob() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv1_global2_with_oob() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: 2 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_oob_in_q() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 2,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv2_with_oob() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 2,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
                partition_size,
                stage_size,
            };
            let problem = AttentionProblem {
                batch: 1,
                num_heads: 1,
                seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize + 9,
                head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_stage2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: 2 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_stage4() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: 4 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_stage2_problem4() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: 2 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_stage2_partition_all2() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 2,
                seq_kv: 2,
                head_dim: 2,
                val_dim: 2,
            };
            let stage_size = AttentionStageSize {
                seq_q: 2 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_reuse_key_value() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 2,
                val_dim: 2,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                TestOptions {
                    reuse_key_value: true,
                    ..Default::default()
                },
            );
        }

        #[test]
        fn attention_double_row_wise() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                TestOptions {
                    two_rows_in_array_tile: true,
                    ..Default::default()
                },
            );
        }

        #[test]
        fn attention_one_tile_masked() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_one_tile_causal() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                causal: true,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_one_tile_masked_causal() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                masked: true,
                causal: true,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_masked_oob() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
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
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_masked_larger() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
                partition_size,
                stage_size,
            };
            let problem = AttentionProblem {
                batch: 1,
                num_heads: 1,
                seq_q: tiling_scheme.elements_in_stage_seq_q() as usize,
                seq_kv: tiling_scheme.elements_in_partition_seq_kv() as usize * 2,
                head_dim: tiling_scheme.elements_in_partition_head_dim() as usize,
                val_dim: tiling_scheme.elements_in_partition_val_dim() as usize,
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }
    };
}
