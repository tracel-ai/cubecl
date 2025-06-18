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
