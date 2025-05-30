#[macro_export]
macro_rules! testgen_matmul_plane_register_algorithm {
    () => {
        use $crate::matmul::kernels::matmul::simple_plane_register::SimplePlaneRegisterAlgorithm;

        mod simple {
            use super::*;

            $crate::testgen_matmul_plane_register_precision!(SimplePlaneRegisterAlgorithm);
        }
    };
}
