#[macro_export]
macro_rules! testgen_matmul_tma_tile {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::TileSize;

        mod t16x16x16 {
            use super::*;

            $crate::testgen_matmul_tma_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize {
                    m: 16,
                    n: 16,
                    k: 16
                })
            );
        }
    };
}
