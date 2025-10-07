mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_plane_mma {
    () => {
        mod matmul_plane_mma {
            use super::*;
            use cubecl_matmul::components::tile::io::Filled;
            type TMM = $crate::components::tile::mma::MmaMatmul<Filled>;

            #[cfg(feature = "matmul_tests_plane")]
            $crate::testgen_matmul_plane_mma_algorithm!();
        }
    };
}
