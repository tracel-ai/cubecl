mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_tma {
    () => {
        mod matmul_tma {
            use super::*;
            use cubecl_matmul::components::tile::io::Filled;
            type TMM = $crate::components::tile::cmma::CmmaMatmul<Filled>;

            #[cfg(feature = "matmul_tests_tma")]
            $crate::testgen_matmul_tma_algorithm!();
        }
    };
}
