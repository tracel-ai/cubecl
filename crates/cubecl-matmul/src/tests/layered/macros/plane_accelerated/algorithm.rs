#[macro_export]
macro_rules! testgen_matmul_plane_accelerated_algorithm {
    () => {
        use $crate::kernels::layered::{
            simple::{SimpleAlgorithm, SimpleBarrierAlgorithm},
            double_buffering::{CyclicDoubleBufferingAlgorithm, TilewiseDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm},
            ordered_double_buffering::OrderedDoubleBufferingAlgorithm,
            specialized::SpecializedAlgorithm
        };
        use $crate::components::global::read::{
            async_full_cyclic,
            sync_full_strided,
            sync_full_tilewise,
            async_full_cooperative,
            async_partial_cyclic,
            async_partial_strided,
        };
        use $crate::components::stage::{
            ColMajorTilingOrder,
            RowMajorTilingOrder
        };

        #[cfg(all(feature = "matmul_tests_simple", feature="matmul_tests_cyclic"))]
        mod simple_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_simple", feature="matmul_tests_strided"))]
        mod simple_strided {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM, sync_full_strided::SyncFullStridedLoading, sync_full_strided::SyncFullStridedLoading>);
        }

        #[cfg(all(feature = "matmul_tests_simple", feature="matmul_tests_tilewise"))]
        mod simple_tilewise {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM, sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>, sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>>);
        }

        #[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
        mod simple_barrier_cooperative {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_cooperative::AsyncFullCooperativeLoading>);
        }

        #[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
        mod simple_barrier_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_cyclic"))]
        mod double_buffering_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(CyclicDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_tilewise"))]
        mod double_buffering_tilewise {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(TilewiseDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_hybrid"))]
        mod double_buffering_hybrid {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(HybridDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(feature = "matmul_tests_ordered")]
        mod ordered_double_buffering {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(OrderedDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_barrier"))]
        mod specialized_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SpecializedAlgorithm<TMM, async_partial_cyclic::AsyncPartialCyclicLoading<ColMajorTilingOrder>>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_barrier"))]
        mod specialized_strided {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SpecializedAlgorithm<TMM, async_partial_strided::AsyncPartialStridedLoading>);
        }
    };
}
