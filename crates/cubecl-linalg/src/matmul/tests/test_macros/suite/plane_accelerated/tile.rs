#[macro_export]
macro_rules! testgen_matmul_accelerated_tile {
    ($algorithm: ty, $precision: ty) => {
        use $crate::matmul::components::TileSize;

        #[cfg(target_os = "macos")]
        mod tl8x8x8 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_shape!(
                $algorithm,
                $precision,
                TileSize { m: 8, n: 8, k: 8 }
            );
        }

        #[cfg(not(target_os = "macos"))]
        mod tl16x16x16 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_shape!(
                $algorithm,
                $precision,
                TileSize {
                    m: 16,
                    n: 16,
                    k: 16
                }
            );
        }

        #[cfg(not(target_os = "macos"))]
        mod tl32x8x16 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_shape!(
                $algorithm,
                $precision,
                TileSize { m: 32, n: 8, k: 16 }
            );
        }

        #[cfg(not(target_os = "macos"))]
        mod tl8x32x16 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_shape!(
                $algorithm,
                $precision,
                TileSize { m: 8, n: 32, k: 16 }
            );
        }

        #[cfg(not(target_os = "macos"))]
        mod tl16x16x8 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_shape!(
                $algorithm,
                $precision,
                TileSize { m: 16, n: 16, k: 8 }
            );
        }
    };
}
