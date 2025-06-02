#[macro_export]
macro_rules! testgen_matmul_unit_partition {
    ($algorithm: ty, $precision: ty, $tile: expr) => {
        use $crate::matmul::components::PartitionSize;

        mod p1x1x4 {
            use super::*;

            $crate::testgen_matmul_unit_stage!(
                $algorithm,
                $precision,
                $tile,
                PartitionSize { m: 1, n: 1, k: 4 }
            );
        }

        mod p2x2x4 {
            use super::*;

            $crate::testgen_matmul_unit_stage!(
                $algorithm,
                $precision,
                $tile,
                PartitionSize { m: 2, n: 2, k: 4 }
            );
        }
    };
}
