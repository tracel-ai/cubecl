#[macro_export]
macro_rules! testgen_matmul_unit_tile {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::TileSize;

        mod t1x1x1 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 1, n: 1, k: 1 })
            );
        }

        mod t8x1x4 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 8, n: 1, k: 4 })
            );
        }

        mod t2x4x1 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 2, n: 4, k: 1 })
            );
        }

        mod t1x8x8 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 1, n: 8, k: 8 })
            );
        }

        mod t4x4x4 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 4, n: 4, k: 4 })
            );
        }

        mod t8x8x8 {
            use super::*;

            $crate::testgen_matmul_unit_partition!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_tile_size(TileSize { m: 8, n: 8, k: 8 })
            );
        }
    };
}
