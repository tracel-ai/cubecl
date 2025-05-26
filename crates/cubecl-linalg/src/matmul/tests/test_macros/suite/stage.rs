/// Select stage size (s)
#[macro_export]
macro_rules! testgen_matmul_stage {
    ($kind: ident, $algorithm: ty, $precision: ty, $layout_lhs: ident, $layout_rhs: ident, $tile: expr) => {
        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_problem!(
                $kind,
                $algorithm,
                $precision,
                $layout_lhs,
                $layout_rhs,
                $tile,
                MatmulSize { m: 1, n: 1, k: 1 }
            );
        }

        // mod s8x8x1 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 8, n: 8, k: 1 }
        //     );
        // }

        // #[cfg(target_os = "macos")]
        // mod s16x16x1 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 16, n: 16, k: 1 }
        //     );
        // }

        // mod s2x2x2 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 2, n: 2, k: 2 }
        //     );
        // }

        // #[cfg(target_os = "macos")]
        // mod s8x8x4 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 8, n: 8, k: 4 }
        //     );
        // }

        // #[cfg(target_os = "macos")]
        // mod s16x8x4 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 16, n: 8, k: 4 }
        //     );
        // }

        // #[cfg(not(target_os = "macos"))]
        // mod s4x4x2 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 4, n: 4, k: 2 }
        //     );
        // }

        // #[cfg(not(target_os = "macos"))]
        // mod s8x4x2 {
        //     use super::*;
        //     $crate::testgen_matmul_problem!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         $tile,
        //         MatmulSize { m: 8, n: 4, k: 2 }
        //     );
        // }
    };
}
