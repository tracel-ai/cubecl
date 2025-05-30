#[macro_export]
macro_rules! testgen_matmul_plane_register_precision {
    ($algorithm: ty) => {
        mod f16_ty {
            use super::*;

            $crate::testgen_matmul_plane_register_tile!($algorithm, (half::f16, half::f16));
        }

        mod f32_ty {
            use super::*;

            $crate::testgen_matmul_plane_register_tile!($algorithm, (f32, f32));
        }
    };
}
