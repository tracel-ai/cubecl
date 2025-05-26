#[macro_export]
macro_rules! testgen_matmul_accelerated_precision {
    ($algorithm: ty) => {
        mod f16_ty {
            use super::*;

            $crate::testgen_matmul_accelerated_tile!(
                $algorithm,
                (half::f16, half::f16),
                NotConstrained
            );
        }

        mod f32_ty {
            use super::*;

            $crate::testgen_matmul_accelerated_tile!($algorithm, (f32, f32), Constrained);
        }
    };
}
