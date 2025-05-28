#[macro_export]
macro_rules! testgen_matmul_stage_k {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr) => {
        mod sk1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                1
            );
        }

        mod sk2 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                2
            );
        }

        mod sk4 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                4
            );
        }
    };
}
