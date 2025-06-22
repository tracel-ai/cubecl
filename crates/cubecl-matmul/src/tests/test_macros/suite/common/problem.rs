#[macro_export]
macro_rules! testgen_matmul_problem {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr, $specialized:expr, $layouts: expr) => {
        use $crate::components::MatmulProblemSize;

        mod g256x256x256 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                $layouts,
                MatmulProblemSize {
                    m: 256,
                    n: 256,
                    k: 256
                }
            );
        }

        #[cfg(feature = "matmul_tests_alt_shapes")]
        mod g100x100x100 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                $layouts,
                MatmulProblemSize {
                    m: 100,
                    n: 100,
                    k: 100
                }
            );
        }

        #[cfg(feature = "matmul_tests_alt_shapes")]
        mod g23x1x17 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                $layouts,
                MatmulProblemSize { m: 23, n: 1, k: 17 }
            );
        }
    };
}
