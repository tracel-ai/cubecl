#[macro_export]
macro_rules! testgen_matmul_unit_partition {
    ($algorithm: ty, $precision: ty, $tile: expr) => {
        use $crate::matmul::components::stage::TilesPerPartition;

        mod pt1x1 {
            use super::*;

            $crate::testgen_matmul_unit_stage!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 1, n: 1 }
            );
        }

        mod pt2x2 {
            use super::*;

            $crate::testgen_matmul_unit_stage!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 2, n: 2 }
            );
        }
    };
}
