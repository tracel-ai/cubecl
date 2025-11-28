#[macro_export]
macro_rules! testgen_attention_suite {
    ($precision: ty) => {
        use super::*;

        #[test]
        fn attention_one_tile_simple() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_one_partition_several_planes() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: STAGE_Q_BASE * 2,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
                partition_size,
                stage_size,
            };
            let problem = AttentionProblem {
                batch: 1,
                num_heads: 1,
                seq_q: elements_in_partition_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_problem_smaller_than_one_tile_seq_q_seq_kv_val_dim() {
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
                seq_q: tiling_scheme.tile_size.seq_q as usize - 1,
                seq_kv: tiling_scheme.tile_size.seq_kv as usize - 1,
                head_dim: tiling_scheme.tile_size.head_dim as usize,
                val_dim: tiling_scheme.tile_size.val_dim as usize - 1,
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_head_dim_oob() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: tiling_scheme.tile_size.head_dim as usize - 1,
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };

            print_problem_vs_scheme(&problem, &tiling_scheme);

            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_partition_temp() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * num_iterations,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * num_iterations,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) - 1,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_seqq2_global2_kv2_global2() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * 2,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_many_planes() {
            let client = TestRuntime::client(&Default::default());

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: 1,
                val_dim: 1,
            };
            let stage_size = AttentionStageSize {
                seq_q: 15 * STAGE_Q_BASE,
            };
            let tiling_scheme = AttentionTilingScheme {
                tile_size: TILE_SIZE,
                partition_size,
                stage_size,
            };
            let problem = AttentionProblem {
                batch: 1,
                num_heads: 1,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * 2 + 1,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) + 9,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv2_causal() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: true,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            );
        }

        #[test]
        fn attention_partition_kv2_masked() {
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme) * 2,
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * 2,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: true,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: true,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) - 1,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: false,
            };

            attention_test_launch::<Algorithm, $precision, TestRuntime>(
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
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme) * 2,
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_num_heads_2() {
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
                num_heads: 2,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_batch_2() {
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
                batch: 2,
                num_heads: 1,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_batch_2_seqq2() {
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
                batch: 2,
                num_heads: 1,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_num_heads_2_batch_2() {
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
                batch: 2,
                num_heads: 2,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: false,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_num_heads_2_masked() {
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
                num_heads: 2,
                seq_q: elements_in_stage_seq_q(&tiling_scheme),
                seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
                head_dim: elements_in_partition_head_dim(&tiling_scheme),
                val_dim: elements_in_partition_val_dim(&tiling_scheme),
                masked: true,
                causal: false,
            };
            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }

        #[test]
        fn attention_huge_problem() {
            let client = TestRuntime::client(&Default::default());

            let batch = 1;
            let num_heads = 1;
            let seq_q = 128;
            let seq_kv = 128;
            let head_dim = 64;
            let val_dim = 64;

            let partition_size = AttentionPartitionSize {
                seq_q: 1,
                seq_kv: 1,
                head_dim: head_dim as u32 / TILE_SIZE.head_dim,
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
                batch,
                num_heads,
                seq_q,
                seq_kv,
                head_dim,
                val_dim,
                masked: false,
                causal: false,
            };

            attention_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                Default::default(),
            )
        }
    };
}
