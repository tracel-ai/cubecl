use cubecl_core::{Runtime, client::ComputeClient};

mod suite;

use crate::{
    components::{
        AttentionProblem, AttentionSelection, AttentionTilingScheme, batch::HypercubeSelection,
    },
    kernels::Algorithm,
    tests::attention_test_launcher::test_attention_algorithm,
};

#[derive(Default)]
pub struct TestOptions {
    pub reuse_key_value: bool,
    pub two_rows_in_array_tile: bool,
}

pub fn attention_test_launch<A: Algorithm, R: Runtime>(
    client: ComputeClient<R::Server>,
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
        use super::*;

        #[cfg(feature = "attention_tests")]
        mod attention_dummy_register {
            type Algorithm = cubecl_attention::kernels::dummy::DummyRegisterAlgorithm;
            const TILE_SIZE: cubecl_attention::components::AttentionTileSize =
                cubecl_attention::components::AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
            const STAGE_Q_BASE: u32 = 1;

            $crate::testgen_attention_suite!();
        }

        #[cfg(feature = "attention_tests")]
        mod attention_unit {
            type Algorithm = cubecl_attention::kernels::unit::UnitAlgorithm;
            const TILE_SIZE: cubecl_attention::components::AttentionTileSize =
                cubecl_attention::components::AttentionTileSize {
                    seq_q: 4,
                    seq_kv: 4,
                    head_dim: 4,
                    val_dim: 4,
                };
            const STAGE_Q_BASE: u32 = 32;

            $crate::testgen_attention_suite!();
        }

        #[cfg(feature = "attention_tests")]
        mod attention_dummy_accelerated {
            type Algorithm = cubecl_attention::kernels::dummy::DummyAcceleratedAlgorithm;
            #[cfg(target_os = "macos")]
            const TILE_SIZE: cubecl_attention::components::AttentionTileSize =
                cubecl_attention::components::AttentionTileSize {
                    seq_q: 8,
                    seq_kv: 8,
                    head_dim: 8,
                    val_dim: 8,
                };
            #[cfg(not(target_os = "macos"))]
            const TILE_SIZE: cubecl_attention::components::AttentionTileSize =
                cubecl_attention::components::AttentionTileSize {
                    seq_q: 16,
                    seq_kv: 16,
                    head_dim: 16,
                    val_dim: 16,
                };
            const STAGE_Q_BASE: u32 = 1;

            // Deactivated
            // $crate::testgen_attention_suite!();
        }
    };
}
