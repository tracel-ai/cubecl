/// tmxnxk
#[macro_export]
macro_rules! testgen_matmul_tile {
    ($kind: ident, $algorithm: ty, $precision: ty, $layout_lhs: ident, $layout_rhs: ident) => {
        mod t1x1x1 {
            use super::*;

            $crate::testgen_matmul_stage!(
                $kind,
                $algorithm,
                $precision,
                $layout_lhs,
                $layout_rhs,
                MatmulSize { m: 1, n: 1, k: 1 }
            );
        }

        // mod t8x1x4 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 8, n: 1, k: 4 }
        //     );
        // }

        // mod t2x4x1 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 2, n: 4, k: 1 }
        //     );
        // }

        // mod t1x8x8 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 1, n: 8, k: 8 }
        //     );
        // }

        // mod t4x4x4 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 4, n: 4, k: 4 }
        //     );
        // }

        // mod t8x8x8 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 8, n: 8, k: 8 }
        //     );
        // }

        // mod t16x16x16 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize {
        //             m: 16,
        //             n: 16,
        //             k: 16
        //         }
        //     );
        // }

        // mod t32x8x16 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 32, n: 8, k: 16 }
        //     );
        // }

        // mod t8x32x16 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 8, n: 32, k: 16 }
        //     );
        // }

        // mod t16x16x8 {
        //     use super::*;
        //     $crate::testgen_matmul_stage!(
        //         $kind,
        //         $algorithm,
        //         $precision,
        //         $layout_lhs,
        //         $layout_rhs,
        //         MatmulSize { m: 16, n: 16, k: 8 }
        //     );
        // }
    };
}
