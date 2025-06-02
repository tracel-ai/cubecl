#[macro_export]
macro_rules! testgen_matmul_quantized_stage {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr) => {
        use $crate::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Quantized,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 1, n: 1, k: 1 }
            );
        }
    };
}
