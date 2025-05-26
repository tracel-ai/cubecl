#[macro_export]
macro_rules! testgen_matmul_layout {
    ($kind: ident, $algorithm: ty, $precision: ty) => {
        use $crate::matmul::components::{MatmulSize, MatrixLayout};

        mod rr {
            use super::*;

            $crate::testgen_matmul_tile!($kind, $algorithm, $precision, RowMajor, RowMajor);
        }

        // mod rc {
        //     use super::*;

        //     $crate::testgen_matmul_tile!($kind, $algorithm, $precision, RowMajor, ColMajor);
        // }

        // mod cr {
        //     use super::*;

        //     $crate::testgen_matmul_tile!($kind, $algorithm, $precision, ColMajor, RowMajor);
        // }

        // mod cc {
        //     use super::*;

        //     $crate::testgen_matmul_tile!($kind, $algorithm, $precision, ColMajor, ColMajor);
        // }
    };
}
