#[macro_export]
macro_rules! testgen_attention_explicit_suite {
    ($precision: ty) => {
        use super::*;

        macro_rules! eg {
            ($x:expr) => {
                <$precision as TestPrecision>::EG::new($x)
            };
        }

        #[test]
        fn attention_explicit_2_2() {
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

            let query = vec![eg!(1.), eg!(0.), eg!(0.), eg!(1.)];
            let key = vec![eg!(1.), eg!(0.), eg!(0.), eg!(1.)];
            let value = vec![eg!(1.), eg!(2.), eg!(3.), eg!(4.)];

            let problem = AttentionProblem {
                batch: 1,
                num_heads: 1,
                seq_q: 2,
                seq_kv: 2,
                head_dim: 2,
                val_dim: 2,
                masked: false,
                causal: false,
            };

            attention_explicit_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                query,
                key,
                value,
                None,
            );
        }

        #[test]
        fn attention_explicit_4_4() {
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
                seq_kv: 4,
                head_dim: 4,
                val_dim: 4,
                masked: false,
                causal: false,
            };

            let query: Vec<_> = (0..problem.seq_q * problem.head_dim)
                .map(|i| eg!((i % 7) as f32 + 0.1 * (i / 8) as f32))
                .collect();
            let key: Vec<_> = (0..problem.seq_kv * problem.head_dim)
                .map(|i| eg!((i % 6) as f32 + 0.05 * (i / 8) as f32))
                .collect();
            let value: Vec<_> = (0..problem.seq_kv * problem.val_dim)
                .map(|i| eg!((i % 5 + 1) as f32))
                .collect();

            print_problem_vs_scheme(&problem, &tiling_scheme);

            attention_explicit_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                query,
                key,
                value,
                None,
            );
        }

        #[test]
        fn attention_explicit_8_8() {
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
                seq_q: 8,
                seq_kv: 8,
                head_dim: 8,
                val_dim: 8,
                masked: false,
                causal: false,
            };

            let query: Vec<_> = (0..problem.seq_q * problem.head_dim)
                .map(|i| eg!((i % 7) as f32 + 0.1 * (i / 8) as f32))
                .collect();
            let key: Vec<_> = (0..problem.seq_kv * problem.head_dim)
                .map(|i| eg!((i % 6) as f32 + 0.05 * (i / 8) as f32))
                .collect();
            let value: Vec<_> = (0..problem.seq_kv * problem.val_dim)
                .map(|i| eg!((i % 5 + 1) as f32))
                .collect();

            print_problem_vs_scheme(&problem, &tiling_scheme);

            attention_explicit_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                query,
                key,
                value,
                None,
            );
        }

        #[test]
        fn attention_explicit_128_4() {
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
                seq_q: 128,
                seq_kv: 4,
                head_dim: 4,
                val_dim: 4,
                masked: false,
                causal: false,
            };

            let query: Vec<_> = (0..problem.seq_q * problem.head_dim)
                .map(|i| eg!((i % 7) as f32 + 0.1 * (i / 8) as f32))
                .collect();
            let key: Vec<_> = (0..problem.seq_kv * problem.head_dim)
                .map(|i| eg!((i % 6) as f32 + 0.05 * (i / 8) as f32))
                .collect();
            let value: Vec<_> = (0..problem.seq_kv * problem.val_dim)
                .map(|i| eg!((i % 5 + 1) as f32))
                .collect();

            print_problem_vs_scheme(&problem, &tiling_scheme);

            attention_explicit_test_launch::<Algorithm, $precision, TestRuntime>(
                client,
                tiling_scheme,
                problem,
                query,
                key,
                value,
                None,
            );
        }
    };
}
