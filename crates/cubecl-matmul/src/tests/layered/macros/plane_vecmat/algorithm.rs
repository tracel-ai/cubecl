#[macro_export]
macro_rules! testgen_matmul_plane_vecmat_algorithm {
    () => {
        use $crate::kernels::layered::{
            double_buffering::CyclicDoubleBufferingAlgorithm, simple::SimpleAlgorithm,
        };

        #[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_cyclic"))]
        mod simple_cyclic {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_precision!(SimpleAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_cyclic"))]
        mod double_buffering_cyclic {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_precision!(CyclicDoubleBufferingAlgorithm<TMM>);
        }
    };
}
