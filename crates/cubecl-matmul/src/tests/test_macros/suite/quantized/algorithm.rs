#[macro_export]
macro_rules! testgen_matmul_quantized_algorithm {
    () => {
        mod simple_cyclic {
            use super::*;
            use $crate::kernels::matmul::simple::SimpleAlgorithm;

            $crate::testgen_matmul_quantized_precision!(SimpleAlgorithm<TMM>);
        }
    };
}
