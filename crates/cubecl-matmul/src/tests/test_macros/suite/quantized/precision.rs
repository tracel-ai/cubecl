#[macro_export]
macro_rules! testgen_matmul_quantized_precision {
    ($algorithm: ty) => {
        mod sym_q8 {
            use super::*;

            $crate::testgen_matmul_quantized_tile!($algorithm, $crate::matmul::tests::SymQ8);
        }
    };
}
