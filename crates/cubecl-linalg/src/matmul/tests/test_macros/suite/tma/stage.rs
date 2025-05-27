#[macro_export]
macro_rules! testgen_matmul_tma_stage {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition: expr) => {
        mod sg1x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 1, n: 1, k: 1 }
            );
        }

        mod sg8x8x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 8, n: 8, k: 1 }
            );
        }

        mod sg16x16x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 16, n: 16, k: 1 }
            );
        }

        mod sg2x2x2 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 2, n: 2, k: 2 }
            );
        }

        mod sg8x8x4 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 8, n: 8, k: 4 }
            );
        }

        mod sg16x8x4 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 16, n: 8, k: 4 }
            );
        }

        mod sg4x4x2 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 4, n: 4, k: 2 }
            );
        }

        mod sg8x4x2 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition,
                MatmulSize { m: 8, n: 4, k: 2 }
            );
        }
    };
}
