#[macro_export]
macro_rules! testgen_matmul_layouts {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr, $specialized: expr) => {
        use $crate::components::MatrixLayout;

        mod rr {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod rc {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod cr {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod cc {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                $specialized,
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
            );
        }
    };
}
