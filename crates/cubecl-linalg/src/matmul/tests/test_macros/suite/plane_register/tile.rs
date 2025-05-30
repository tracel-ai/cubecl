#[macro_export]
macro_rules! testgen_matmul_plane_register_tile {
    ($algorithm: ty, $precision: ty) => {
        use $crate::matmul::components::TileSize;

        mod t32x32x32 {
            use super::*;

            $crate::testgen_matmul_plane_register_partition!(
                $algorithm,
                $precision,
                TileSize {
                    m: 32,
                    n: 32,
                    k: 32
                }
            );
        }

        mod t1x32x32 {
            use super::*;

            $crate::testgen_matmul_plane_register_partition!(
                $algorithm,
                $precision,
                TileSize { m: 1, n: 32, k: 32 }
            );
        }

        mod t32x1x32 {
            use super::*;

            $crate::testgen_matmul_plane_register_partition!(
                $algorithm,
                $precision,
                TileSize { m: 32, n: 1, k: 32 }
            );
        }

        mod t32x32x1 {
            use super::*;

            $crate::testgen_matmul_plane_register_partition!(
                $algorithm,
                $precision,
                TileSize { m: 32, n: 32, k: 1 }
            );
        }
    };
}
