#[macro_export]
macro_rules! testgen_matmul_accelerated_stage {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr) => {
        use $crate::matmul::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 1, n: 1, k: 1 }
            );
        }

        mod s2x2x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 2, n: 2, k: 1 }
            );
        }

        mod s4x4x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 4, n: 4, k: 1 }
            );
        }

        mod s8x4x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 8, n: 4, k: 1 }
            );
        }

        mod s8x8x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 8, n: 8, k: 1 }
            );
        }
    };
}
