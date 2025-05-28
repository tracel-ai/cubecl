#[macro_export]
macro_rules! testgen_matmul_layouts {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage_k: expr) => {
        use $crate::matmul::components::{MatmulSize, MatrixLayout};

        mod rr {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                $partition_count,
                $stage_k,
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
                $partition_shape,
                $partition_count,
                $stage_k,
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
                $partition_shape,
                $partition_count,
                $stage_k,
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
                $partition_shape,
                $partition_count,
                $stage_k,
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
            );
        }
    };
}
