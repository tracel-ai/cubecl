#[macro_export]
macro_rules! testgen_matmul_tma_partition {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::PartitionSize;

        mod p1x1x4 {
            use super::*;

            $crate::testgen_matmul_tma_stage!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 4 })
            );
        }
    };
}
