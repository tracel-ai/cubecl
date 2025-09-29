mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_plane_accelerated {
    () => {
        mod matmul_plane_accelerated {
            use super::*;
            use cubecl_matmul::components::tile::io::Filled;
            type TMM = $crate::components::tile::accelerated::AcceleratedMatmul<Filled>;

            #[cfg(feature = "matmul_tests_plane")]
            $crate::testgen_matmul_plane_accelerated_algorithm!();
        }
    };
}
