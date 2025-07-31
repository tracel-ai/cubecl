use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::TileSize;

use crate::{
    components::{AttentionProblem, AttentionSelection, batch::HypercubeSelection},
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_first_test<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let problem = AttentionProblem {
        batch: 1,
        seq_q: 8,
        seq_k: 8,
        num_heads: 1,
        head_dim: 8,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        score_tile_size: TileSize { m: 8, n: 8, k: 8 },
        value_tile_size: TileSize { m: 8, n: 8, k: 8 },
        plane_dim: 32,
    };

    test_attention_algorithm::<DummyAlgorithm, (f32, f32), R>(client, problem, selection);
}

#[macro_export]
macro_rules! testgen_attention {
    () => {
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
