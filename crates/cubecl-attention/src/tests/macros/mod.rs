use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionProblem, AttentionSelection, batch::HypercubeSelection,
        tile::dummy::AttentionTileSize,
    },
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_tile_wide<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    attention_tile_size: AttentionTileSize,
) {
    assert!(attention_tile_size.head_dim == attention_tile_size.val_dim);

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q: attention_tile_size.seq_q as usize,
        seq_kv: attention_tile_size.seq_kv as usize,
        head_dim: attention_tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        attention_tile_size,
        plane_dim: 32,
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

pub fn attention_longer_seq_kv<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    attention_tile_size: AttentionTileSize,
    seq_kv: usize,
) {
    assert!(attention_tile_size.head_dim == attention_tile_size.val_dim);

    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q: attention_tile_size.seq_q as usize,
        seq_kv,
        head_dim: attention_tile_size.head_dim as usize,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        attention_tile_size,
        plane_dim: 32,
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

#[macro_export]
macro_rules! testgen_attention {
    () => {
        #[cfg(feature = "attention_tests")]
        mod attention {
            use super::*;
            use cubecl_attention::components::tile::dummy::AttentionTileSize;

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
            fn attention_8_64() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_longer_seq_kv::<TestRuntime>(
                    client,
                    attention_tile_size,
                    64,
                )
            }

            #[test]
            fn attention_8_58() {
                let client = TestRuntime::client(&Default::default());
                let attention_tile_size = AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
                $crate::tests::macros::attention_longer_seq_kv::<TestRuntime>(
                    client,
                    attention_tile_size,
                    58,
                )
            }
        }
    };
}
