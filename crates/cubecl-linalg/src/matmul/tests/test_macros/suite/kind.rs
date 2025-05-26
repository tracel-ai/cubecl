#[macro_export]
macro_rules! testgen_matmul_plane_accelerated {
    () => {
        mod matmul_plane_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            $crate::testgen_matmul_algorithm!(Accelerated);
        }
    };
}

// #[macro_export]
// macro_rules! testgen_matmul_unit {
//     () => {
//         mod matmul_unit {
//             type TMM = $crate::matmul::components::tile::register_matmul::RegisterMatmul;
//             $crate::testgen_matmul_algorithm!(Unit);
//         }
//     };
// }

// #[macro_export]
// macro_rules! testgen_matmul_tma {
//     () => {
//         mod matmul_tma {
//             type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;
//             $crate::testgen_matmul_algorithm!(Tma);
//         }
//     };
// }

// #[macro_export]
// macro_rules! testgen_matmul_quantized {
//     () => {
//         mod matmul_quantized {
//             type Precision = $crate::matmul::tests::SymQ8;
//             type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;
//             $crate::testgen_matmul_algorithm!(Quantized);
//         }
//     };
// }
