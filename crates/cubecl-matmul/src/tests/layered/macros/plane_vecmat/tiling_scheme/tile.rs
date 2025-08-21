#[macro_export]
macro_rules! testgen_matmul_plane_vecmat_tile {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::TileSize;

        // For vec 8
        mod t1x8x256 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 1, n: 8, k: 256 })
            );
        }

        // For vec 4
        mod t1x4x128 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 1, n: 4, k: 128 })
            );
        }

        // For vec 1
        mod t1x1x128 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 1, n: 1, k: 128 })
            );
        }
    };
}
