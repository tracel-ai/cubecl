#[macro_export]
macro_rules! testgen_matmul_tma_algorithm {
    () => {
        mod simple_tma {
            use super::*;
            use $crate::kernels::layered::simple::SimpleTmaAlgorithm;

            $crate::testgen_matmul_tma_precision!(SimpleTmaAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double"))]
        mod double_buffering_tma {
            use super::*;
            use $crate::kernels::layered::double_buffering::TmaDoubleBufferingAlgorithm;

            $crate::testgen_matmul_tma_precision!(TmaDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double"))]
        mod specialized_tma {
            use super::*;
            use $crate::kernels::layered::specialized::TmaSpecializedAlgorithm;

            $crate::testgen_matmul_tma_precision!(TmaSpecializedAlgorithm<TMM>);
        }

        // Not yet properly implemented in the macro, just manually test with the hardcoded swizzling
        // patch in the test launcher

        // #[cfg(all(feature = "matmul_tests_double"))]
        // mod specialized_tma_swizzled {
        //     use super::*;
        //     use $crate::components::tile::mma::SwizzledMmaMatmul;
        //     use $crate::kernels::layered::specialized::TmaSwizzledSpecializedAlgorithm;

        //     $crate::testgen_matmul_tma_precision!(
        //         TmaSwizzledSpecializedAlgorithm<SwizzledMmaMatmul>
        //     );
        // }
    };
}
