#[macro_export]
macro_rules! testgen_matmul_tma_tile {
    ($algorithm: ty, $precision: ty) => {
        use $crate::matmul::components::TileSize;

        mod tl16x16x16 {
            use super::*;

            $crate::testgen_matmul_tma_partition_shape!(
                $algorithm,
                $precision,
                TileSize {
                    m: 16,
                    n: 16,
                    k: 16
                }
            );
        }
    };
}
