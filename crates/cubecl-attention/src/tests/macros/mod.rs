use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionPartitionSize, AttentionProblem, AttentionSelection, AttentionTileSize,
        AttentionTilingScheme, batch::HypercubeSelection,
    },
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_tile_wide<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    tile_size: AttentionTileSize,
) {
    assert!(tile_size.head_dim == tile_size.val_dim);

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q: tile_size.seq_q as usize,
        seq_kv: tile_size.seq_kv as usize,
        head_dim: tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme: AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: 1,
                seq_kv: 1,
                val_dim: 1,
            },
        },
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

pub fn attention_different_seq_kv<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    tile_size: AttentionTileSize,
    seq_kv: usize,
) {
    assert!(tile_size.head_dim == tile_size.val_dim);

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q: tile_size.seq_q as usize,
        seq_kv,
        head_dim: tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme: AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: 1,
                seq_kv: 1,
                val_dim: 1,
            },
        },
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

pub fn attention_different_seq_q<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    tile_size: AttentionTileSize,
    seq_q: usize,
) {
    assert!(tile_size.head_dim == tile_size.val_dim);

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q,
        seq_kv: tile_size.seq_kv as usize,
        head_dim: tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme: AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: 1,
                seq_kv: 1,
                val_dim: 1,
            },
        },
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

pub fn attention_larger_partition<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    partition_size: AttentionPartitionSize,
    seq_q: usize,
) {
    let tile_size = AttentionTileSize {
        seq_q: 8,
        seq_kv: 8,
        head_dim: 8,
        val_dim: 8,
    };

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q,
        seq_kv: tile_size.seq_kv as usize,
        head_dim: tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        plane_dim: 32,
        tiling_scheme: AttentionTilingScheme {
            tile_size,
            partition_size,
        },
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

#[macro_export]
macro_rules! testgen_attention {
    () => {
        #[cfg(feature = "attention_tests")]
        mod attention {
            use super::*;
            use cubecl_attention::components::AttentionTileSize;

            #[test]
            fn attention_8_8_8_8() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_tile_wide::<TestRuntime>(
                    client,
                    attention_tile_size,
                )
            }

            #[test]
            fn attention_9_9_9_9() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 9,
                    seq_kv: 9,
                    head_dim: 9,
                    val_dim: 9,
                };
                $crate::tests::macros::attention_tile_wide::<TestRuntime>(
                    client,
                    attention_tile_size,
                )
            }

            #[test]
            fn attention_7_3_10_10() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 7,
                    seq_kv: 3,
                    head_dim: 10,
                    val_dim: 10,
                };
                $crate::tests::macros::attention_tile_wide::<TestRuntime>(
                    client,
                    attention_tile_size,
                )
            }

            #[test]
            fn attention_8_kv64() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_different_seq_kv::<TestRuntime>(
                    client,
                    attention_tile_size,
                    64,
                )
            }

            #[test]
            fn attention_8_kv58() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_different_seq_kv::<TestRuntime>(
                    client,
                    attention_tile_size,
                    58,
                )
            }

            #[test]
            fn attention_8_kv5() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_different_seq_kv::<TestRuntime>(
                    client,
                    attention_tile_size,
                    5,
                )
            }

            #[test]
            fn attention_8_q16() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_different_seq_q::<TestRuntime>(
                    client,
                    attention_tile_size,
                    16,
                )
            }

            #[test]
            fn attention_8_q4() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_different_seq_q::<TestRuntime>(
                    client,
                    attention_tile_size,
                    4,
                )
            }
        }
    };
}
