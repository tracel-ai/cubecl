use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        AttentionBlueprint, AttentionElems, AttentionIdent, AttentionProblem,
        AttentionTilingScheme, AvailableLineSizes, batch::HypercubeBlueprint,
    },
    kernels::{Algorithm, SharedAttentionSettings},
    tests::{attention_test_launcher::test_attention_algorithm, test_utils::TestPrecision},
};

mod suite;

pub mod tiling_scheme_ops {
    use crate::components::{AttentionProblem, AttentionTilingScheme};

    pub fn elements_in_stage_seq_q(tiling_scheme: &AttentionTilingScheme) -> usize {
        tiling_scheme.stage_size.seq_q as usize * elements_in_partition_seq_q(tiling_scheme)
    }

    pub fn elements_in_partition_seq_q(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.seq_q * tiling_scheme.partition_size.seq_q) as usize
    }

    pub fn elements_in_partition_head_dim(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.head_dim * tiling_scheme.partition_size.head_dim) as usize
    }

    pub fn elements_in_partition_seq_kv(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.seq_kv * tiling_scheme.partition_size.seq_kv) as usize
    }

    pub fn elements_in_partition_val_dim(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.val_dim * tiling_scheme.partition_size.val_dim) as usize
    }

    pub fn print_problem_vs_scheme(
        problem: &AttentionProblem,
        tiling_scheme: &AttentionTilingScheme,
    ) {
        println!(
            "seq_q: problem {:?} vs scheme {:?}",
            problem.seq_q,
            elements_in_stage_seq_q(&tiling_scheme),
        );
        println!(
            "seq_kv: problem {:?} vs scheme {:?}",
            problem.seq_kv,
            elements_in_partition_seq_kv(&tiling_scheme)
        );
        println!(
            "head_dim: problem {:?} vs scheme {:?}",
            problem.head_dim,
            elements_in_partition_head_dim(&tiling_scheme)
        );
        println!(
            "val_dim: problem {:?} vs scheme {:?}",
            problem.val_dim,
            elements_in_partition_val_dim(&tiling_scheme)
        );
    }
}

pub fn attention_test_launch<A: Algorithm, P: TestPrecision, R: Runtime>(
    client: ComputeClient<R>,
    problem: AttentionProblem,
    settings: &A::Settings,
) {
    // let attention_elems = AttentionElems::new::<P::AP>();
    // let line_sizes = {
    //     let ls = AvailableLineSizes::from_elem_types(
    //         &client,
    //         attention_elems.query_global.size(),
    //         attention_elems.mask.size(),
    //         attention_elems.out_global.size(),
    //     );
    //     let ls = A::filter_line_sizes(ls)
    //         .filter_with_tensor(AttentionIdent::Query, &query.strides, &query.shape)
    //         .filter_with_tensor(AttentionIdent::Key, &key.strides, &key.shape)
    //         .filter_with_tensor(AttentionIdent::Value, &value.strides, &value.shape)
    //         .filter_with_tensor(AttentionIdent::Out, &out.strides, &out.shape);

    //     if let Some(mask) = mask.as_ref() {
    //         ls.filter_with_tensor(AttentionIdent::Mask, &mask.strides, &mask.shape)
    //     } else {
    //         ls
    //     }
    // }
    // .pick_max()
    // .unwrap();

    // let blueprint = AttentionBlueprint {
    //     hypercube_blueprint: HypercubeBlueprint {},
    //     plane_dim: 32,
    //     tiling_scheme,
    //     reuse_key_value: test_options.reuse_key_value,
    //     two_rows_in_array_tile: test_options.two_rows_in_array_tile,
    //     line_sizes: problem.line_sizes.clone(),
    //     masked: problem.masked,
    //     causal: problem.causal,
    //     check_bounds: tiling_scheme.check_bounds(&problem),
    //     num_planes: todo!(),
    // };
    let blueprint = A::blueprint(&client, &problem, settings).unwrap();

    test_attention_algorithm::<A, P, R>(client, problem, blueprint);
}

#[macro_export]
macro_rules! testgen_attention {
    () => {
        use super::*;

        #[cfg(feature = "attention_tests_unit")]
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

            $crate::testgen_attention_precision!();
        }

        #[cfg(feature = "attention_tests_blackbox_accelerated")]
        mod attention_blackbox_accelerated {
            type Algorithm =
                cubecl_attention::kernels::blackbox_accelerated::BlackboxAcceleratedAlgorithm;
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

            $crate::testgen_attention_precision!();
        }
    };
}

#[macro_export]
macro_rules! testgen_attention_precision {
    () => {
        use super::*;

        use cubecl_attention::components::{
            AccumulatorPrecision, AttentionIdent, AttentionLineSizes, AttentionPartitionSize,
            AttentionProblem, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
            AvailableLineSizes,
        };
        use $crate::tests::macros::{attention_test_launch, tiling_scheme_ops::*};

        use $crate::tests::TestPrecision;

        #[cfg(feature = "attention_tests_f16")]
        mod f16_ty {
            use super::*;

            $crate::testgen_attention_suite!((half::f16, half::f16));
        }

        #[cfg(feature = "attention_tests_f32")]
        mod f32_ty {
            use super::*;

            $crate::testgen_attention_suite!((f32, f32));
        }
    };
}
