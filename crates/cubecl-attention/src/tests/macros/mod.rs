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
            elements_in_stage_seq_q(tiling_scheme),
        );
        println!(
            "seq_kv: problem {:?} vs scheme {:?}",
            problem.seq_kv,
            elements_in_partition_seq_kv(tiling_scheme)
        );
        println!(
            "head_dim: problem {:?} vs scheme {:?}",
            problem.head_dim,
            elements_in_partition_head_dim(tiling_scheme)
        );
        println!(
            "val_dim: problem {:?} vs scheme {:?}",
            problem.val_dim,
            elements_in_partition_val_dim(tiling_scheme)
        );
    }
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
            AttentionProblem, AttentionStageSize, AttentionStorageTypes, AttentionTileSize,
            AttentionTilingScheme, AvailableLineSizes,
        };
        use $crate::kernels::SharedAttentionSettings;
        use $crate::tests::attention_test_launcher::attention_test_launch;
        use $crate::tests::macros::tiling_scheme_ops::*;

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
