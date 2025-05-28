#[macro_export]
macro_rules! testgen_matmul_unit_tile {
    ($algorithm: ty, $precision: ty) => {
        use $crate::matmul::components::TileShape;

        mod tl1x1x1 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 1, n: 1, k: 1 }
            );
        }

        mod tl8x1x4 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 8, n: 1, k: 4 }
            );
        }

        mod tl2x4x1 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 2, n: 4, k: 1 }
            );
        }

        mod tl1x8x8 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 1, n: 8, k: 8 }
            );
        }

        mod tl4x4x4 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 4, n: 4, k: 4 }
            );
        }

        mod tl8x8x8 {
            use super::*;

            $crate::testgen_matmul_unit_partition_shape!(
                $algorithm,
                $precision,
                TileShape { m: 8, n: 8, k: 8 }
            );
        }
    };
}
