#[macro_export]
macro_rules! testgen_matmul_plane_accelerated_algorithm {
    () => {
        use $crate::kernels::layered::{
            simple::SimpleAlgorithm,
            simple_barrier::SimpleBarrierAlgorithm,
            double_buffering::{CyclicDoubleBufferingAlgorithm, TilewiseDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm},
            ordered_double_buffering::OrderedDoubleBufferingAlgorithm
        };
        use $crate::components::global::read::{
            async_full_cyclic,
            async_full_maximize_slice_length,
            async_full_maximize_unit_count,
            sync_full_strided,
            sync_full_tilewise,
            async_full_cooperative,
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

        #[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
        mod simple_barrier_maximize_slice_length {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_maximize_slice_length::AsyncFullMaximizeSliceLengthLoading>);
        }

        #[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
        mod simple_barrier_maximize_unit_count {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_maximize_unit_count::AsyncFullMaximizeUnitCountLoading>);
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
    };
}
