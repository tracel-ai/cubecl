use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionProblem, AttentionSelection, batch::HypercubeSelection,
        tile::dummy::AttentionTileSize,
    },
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_first_test<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let problem = AttentionProblem {
        batch: 1,
        num_heads: 1,
        seq_q: 8,
        seq_kv: 8,
        head_dim: 8,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        attention_tile_size: AttentionTileSize {
            seq_q: 8,
            head_dim: 8,
            seq_kv: 8,
            val_dim: 8,
        },
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

            #[test]
            fn attention_tmp() {
                let client = TestRuntime::client(&Default::default());
                $crate::tests::macros::attention_first_test::<TestRuntime>(client)
            }
        }
    };
}
