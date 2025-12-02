#[macro_export]
macro_rules! testgen_convolution_accelerated_algorithm {
    () => {
        use $crate::kernels::layered::{
            simple::SimpleConvAlgorithm,
        };
        use $crate::components::global::read::strategy::{
            async_full_cyclic,
            async_full_strided,
        };
        use cubecl_matmul::components::global::read::{
            sync_full_cyclic,
            sync_full_strided,
            sync_full_tilewise,
        };
        use cubecl_matmul::components::stage::{
            ColMajorTilingOrder,
            RowMajorTilingOrder
        };

        #[cfg(all(feature = "conv_tests_simple", feature="conv_tests_cyclic"))]
        mod simple_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleConvAlgorithm<TMM>);
        }

        #[cfg(all(feature = "conv_tests_simple", feature="conv_tests_strided"))]
        mod simple_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleConvAlgorithm<
                TMM,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading
            >);
        }

        #[cfg(all(feature = "conv_tests_simple", feature="conv_tests_tilewise"))]
        mod simple_tilewise {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleConvAlgorithm<
                TMM,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>
            >);
        }

        #[cfg(all(feature = "conv_tests_simple", feature="conv_tests_cyclic", feature = "conv_tests_async_copy"))]
        mod simple_async_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleConvAlgorithm<
                TMM,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>
            >);
        }

        #[cfg(all(feature = "conv_tests_simple", feature="conv_tests_strided", feature = "conv_tests_async_copy"))]
        mod simple_async_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleConvAlgorithm<
                TMM,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading
            >);
        }
    };
}
