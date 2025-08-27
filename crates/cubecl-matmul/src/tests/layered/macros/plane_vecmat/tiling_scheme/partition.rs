#[macro_export]
macro_rules! testgen_matmul_plane_vecmat_partition {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::PartitionSize;

        mod p1x1x1 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_stage!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
            );
        }

        mod p1x2x1 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_stage!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_partition_size(PartitionSize { m: 1, n: 2, k: 1 })
            );
        }

        mod p1x1x2 {
            use super::*;

            $crate::testgen_matmul_plane_vecmat_stage!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 2 })
            );
        }
    };
}
