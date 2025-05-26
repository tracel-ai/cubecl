#[macro_export]
macro_rules! testgen_matmul_tma_algorithm {
    () => {
        mod simple_tma {
            use super::*;
            use $crate::matmul::kernels::matmul::simple_tma::SimpleTmaAlgorithm;

            $crate::testgen_matmul_tma_precision!(SimpleTmaAlgorithm<TMM>);
        }
    };
}
