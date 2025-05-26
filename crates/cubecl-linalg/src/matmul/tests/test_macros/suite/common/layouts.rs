#[macro_export]
macro_rules! testgen_matmul_layouts {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $stage: expr) => {
        use $crate::matmul::components::{MatmulSize, MatrixLayout};

        mod rr {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $stage,
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
                $stage,
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
                $stage,
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
                $stage,
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
            );
        }
    };
}
