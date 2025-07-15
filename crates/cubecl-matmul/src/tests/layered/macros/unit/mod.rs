mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_unit {
    () => {
        mod matmul_unit {
            use super::*;

            #[cfg(feature = "matmul_tests_unit")]
            $crate::testgen_matmul_unit_algorithm!();
        }
    };
}
