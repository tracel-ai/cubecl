mod algorithm;
mod launch;
mod precision;
mod tiling_scheme;

pub use launch::test_algo;

#[macro_export]
macro_rules! testgen_matmul_plane_accelerated {
    () => {
        mod matmul_plane_accelerated {
            use super::*;
            type TMM = $crate::components::tile::accelerated::AcceleratedMatmul;

            #[cfg(feature = "matmul_tests_plane")]
            $crate::testgen_matmul_plane_accelerated_algorithm!();
        }
    };
}
