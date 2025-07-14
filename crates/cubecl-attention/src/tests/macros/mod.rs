use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{AttentionProblem, AttentionSelection, batch::HypercubeSelection},
    kernels::dummy::DummyAlgorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

pub fn attention_first_test<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let problem = AttentionProblem {
        batch: 1,
        seq_q: 1,
        seq_k: 1,
        num_heads: 1,
        head_dim: 1,
        masked: false,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
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
