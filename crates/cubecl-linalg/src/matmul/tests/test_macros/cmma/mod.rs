#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_matmul_accelerated {
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        $crate::matmul_standard_tests!(standard);
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float, $float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_unit {
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        $crate::matmul_standard_tests!(unit);
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_unit {
            use super::*;
            type TMM = $crate::matmul::components::tile::register_matmul::RegisterMatmul;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_unit!($float, $float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_tma {
    ([$($float:ident: $stage: ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_tma {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_tma!($float, $stage);
                })*
            }
        }
    };
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        $crate::matmul_standard_tests!(tma);
    };
}

#[macro_export]
macro_rules! testgen_matmul_quantized {
    () => {
        #[allow(non_snake_case)]
        mod matmul_quantized {
            use super::*;

            type Precision = $crate::matmul::tests::SymQ8;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            $crate::matmul_standard_tests!(standard);
        }
    };
}
