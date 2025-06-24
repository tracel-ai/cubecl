#[macro_export]
macro_rules! testgen_matmul_layouts {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection: expr) => {
        use $crate::components::MatrixLayout;

        mod rr {
            use super::*;

            $crate::testgen_matmul_problem_size!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod rc {
            use super::*;

            $crate::testgen_matmul_problem_size!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod cr {
            use super::*;

            $crate::testgen_matmul_problem_size!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
            );
        }

        #[cfg(feature = "matmul_tests_layouts")]
        mod cc {
            use super::*;

            $crate::testgen_matmul_problem_size!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
            );
        }
    };
}
