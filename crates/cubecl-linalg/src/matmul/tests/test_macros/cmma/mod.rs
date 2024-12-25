#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_matmul_accelerated {
    ($float:ident) => {
        type EG = $float;
        type ES = half::f16;

        $crate::matmul_standard_tests!();
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated::Accelerated;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float);
                })*
            }
        }
    };
}
#[macro_export]
macro_rules! testgen_matmul_plane {
    ($float:ident) => {
        type EG = $float;
        type ES = $float;

        $crate::matmul_standard_tests!();
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::plane::PlaneMma;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float);
                })*
            }
        }
    };
}
