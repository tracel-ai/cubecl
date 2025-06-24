mod algorithm;
mod launch;
mod partition;
mod precision;
mod stage;
mod tile;

pub use launch::test_algo;

#[macro_export]
macro_rules! testgen_matmul_tma {
    () => {
        mod matmul_tma {
            use super::*;
            type TMM = $crate::components::tile::accelerated::AcceleratedMatmul;

            #[cfg(feature = "matmul_tests")]
            $crate::testgen_matmul_tma_algorithm!();
        }
    };
}
