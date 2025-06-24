mod algorithm;
mod partition;
mod precision;
mod stage;
mod tile;

#[macro_export]
macro_rules! testgen_matmul_quantized {
    () => {
        mod matmul_quantized {
            use super::*;
            type TMM = $crate::components::tile::accelerated::AcceleratedMatmul;

            #[cfg(feature = "matmul_tests")]
            $crate::testgen_matmul_quantized_algorithm!();
        }
    };
}
