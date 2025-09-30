#[macro_export]
macro_rules! testgen_matmul_mma_tile {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::TileSize;

        // AMD f16/bf16
        #[cfg(not(target_os = "macos"))]
        mod t16x16x16 {
            use super::*;

            $crate::testgen_matmul_mma_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize {
                    m: 16,
                    n: 16,
                    k: 16
                })
            );
        }

        // Nvidia f16/bf16
        #[cfg(not(target_os = "macos"))]
        mod t16x8x16 {
            use super::*;

            $crate::testgen_matmul_mma_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 16, n: 8, k: 16 })
            );
        }

        // Nvidia tf32
        #[cfg(not(target_os = "macos"))]
        mod t16x8x8 {
            use super::*;

            $crate::testgen_matmul_mma_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 16, n: 8, k: 8 })
            );
        }
    };
}
