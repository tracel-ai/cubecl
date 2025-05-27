#[macro_export]
macro_rules! testgen_matmul_quantized_partition {
    ($algorithm: ty, $precision: ty, $tile: expr) => {
        use $crate::matmul::components::stage::TilesPerPartition;

        mod pt1x1 {
            use super::*;

            $crate::testgen_matmul_quantized_partition_count!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 1, n: 1 }
            );
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_quantized_partition_count {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr) => {
        use $crate::matmul::components::stage::PartitionsPerStage;

        mod pc1x1 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                Quantized,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 1, n: 1 }
            );
        }
    };
}
