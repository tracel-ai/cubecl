#[macro_export]
macro_rules! testgen_matmul_quantized_partition {
    ($algorithm: ty, $precision: ty, $tile: expr) => {
        use $crate::matmul::components::PartitionSize;

        mod p1x1x4 {
            use super::*;

            $crate::testgen_matmul_quantized_stage!(
                $algorithm,
                $precision,
                $tile,
                PartitionSize { m: 1, n: 1, k: 4 }
            );
        }
    };
}
