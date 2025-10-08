#[macro_export]
macro_rules! testgen_matmul_problem_size {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection: expr, $layouts: expr) => {
        use $crate::components::MatmulProblem;

        mod g256x256x256 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                MatmulProblem {
                    m: 256,
                    n: 256,
                    k: 256,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
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
                $selection,
                MatmulProblem {
                    m: 100,
                    n: 100,
                    k: 100,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
                }
            );
        }

        // line_size_lhs != line_size_rhs
        #[cfg(feature = "matmul_tests_alt_shapes")]
        mod g100x99x100 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                MatmulProblem {
                    m: 100,
                    n: 100,
                    k: 100,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
                }
            );
        }

        // line_size_lhs != line_size_out
        #[cfg(feature = "matmul_tests_alt_shapes")]
        mod g100x100x99 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                MatmulProblem {
                    m: 100,
                    n: 100,
                    k: 100,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
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
                $selection,
                MatmulProblem {
                    m: 23,
                    n: 1,
                    k: 17,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
                }
            );
        }

        #[cfg(feature = "matmul_tests_vecmat")]
        mod g1x256x256 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $selection,
                MatmulProblem {
                    m: 1,
                    n: 256,
                    k: 256,
                    lhs_batches: vec![2],
                    rhs_batches: vec![2],
                    out_batches: vec![2],
                    lhs_layout: $layouts.0,
                    rhs_layout: $layouts.1,
                }
            );
        }
    };
}
