#[macro_export]
macro_rules! testgen_matmul_accelerated_partition {
    ($algorithm: ty, $precision: ty, $tile: expr, $constrained: ident) => {
        use $crate::matmul::components::stage::TilesPerPartition;

        mod pt1x1 {
            use super::*;

            $crate::testgen_matmul_accelerated_stage!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 1, n: 1 },
                $constrained
            );
        }

        mod pt2x1 {
            use super::*;

            $crate::testgen_matmul_accelerated_stage!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 2, n: 1 },
                $constrained
            );
        }
    };
}
