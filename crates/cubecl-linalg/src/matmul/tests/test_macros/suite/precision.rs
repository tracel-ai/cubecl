/// f16, f32, etc
#[macro_export]
macro_rules! testgen_matmul_precision {
    ($kind: ident, $algorithm: ty) => {
        mod f16_ty {
            use super::*;

            $crate::testgen_matmul_layout!($kind, $algorithm, (half::f16, half::f16));
        }

        // mod f32_ty {
        //     use super::*;

        //     $crate::testgen_matmul_layout!($kind, $algorithm, f32);
        // }
    };
}
