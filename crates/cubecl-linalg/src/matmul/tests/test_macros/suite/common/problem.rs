#[macro_export]
macro_rules! testgen_matmul_problem {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage: expr, $layouts: expr) => {
        use $crate::matmul::components::MatmulSize;
        mod g100x100x100 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                $stage,
                $layouts,
                MatmulSize {
                    m: 100,
                    n: 100,
                    k: 100
                }
            );
        }

        mod g23x1x17 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                $stage,
                $layouts,
                MatmulSize { m: 23, n: 1, k: 17 }
            );
        }

        mod g256x256x256 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                $stage,
                $layouts,
                MatmulSize {
                    m: 256,
                    n: 256,
                    k: 256
                }
            );
        }
    };
}
