#[macro_export]
macro_rules! testgen_matmul_layouts {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr) => {
        use $crate::matmul::components::MatrixLayout;

        mod rr {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
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
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
            );
        }
    };
}
