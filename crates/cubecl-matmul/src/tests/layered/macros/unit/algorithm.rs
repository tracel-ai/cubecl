#[macro_export]
macro_rules! testgen_matmul_unit_algorithm {
    () => {
        use $crate::kernels::layered::double_unit::DoubleUnitAlgorithm;
        use $crate::kernels::layered::simple_unit::SimpleUnitAlgorithm;

        #[cfg(feature = "matmul_tests_simple")]
        mod simple {
            use super::*;

            $crate::testgen_matmul_unit_precision!(SimpleUnitAlgorithm);
        }

        #[cfg(feature = "matmul_tests_double")]
        mod double_buffering {
            use super::*;

            $crate::testgen_matmul_unit_precision!(DoubleUnitAlgorithm);
        }
    };
}
