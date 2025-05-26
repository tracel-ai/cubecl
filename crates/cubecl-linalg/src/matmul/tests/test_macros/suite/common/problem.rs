#[macro_export]
macro_rules! testgen_matmul_problem {
    ($kind: ident, $algorithm: ty, $precision: ty, $tile: expr, $stage: expr, $layouts: expr) => {
        mod p100x100x100 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $stage,
                $layouts,
                MatmulSize {
                    m: 100,
                    n: 100,
                    k: 100
                }
            );
        }

        mod p23x1x17 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $stage,
                $layouts,
                MatmulSize { m: 23, n: 1, k: 17 }
            );
        }

        mod p256x256x256 {
            use super::*;
            $crate::testgen_matmul_launch!(
                $kind,
                $algorithm,
                $precision,
                $tile,
                $stage,
                $layouts,
                MatmulSize {
                    m: 256,
                    n: 256,
                    k: 256
                }
            );
        }
    };
}
