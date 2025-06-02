#[macro_export]
macro_rules! testgen_matmul_unit_algorithm {
    () => {
        use $crate::kernels::matmul::simple_unit::SimpleUnitAlgorithm;

        mod simple {
            use super::*;

            $crate::testgen_matmul_unit_precision!(SimpleUnitAlgorithm);
        }
    };
}
