mod algorithm;
mod launch;
mod partition;
mod precision;
mod specialized;
mod stage;
mod tile;

pub use launch::test_algo;

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
