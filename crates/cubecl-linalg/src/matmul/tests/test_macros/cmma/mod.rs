#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_matmul_accelerated {
    ($eg:ty, $es:ty) => {
        type EG = $eg;
        type ES = $es;

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
                    $crate::testgen_matmul_accelerated!($float, half::f16);
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
        mod matmul_plane {
            use super::*;
            type TMM = $crate::matmul::components::tile::plane::PlaneMma;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float, $float);
                })*
            }
        }
    };
}
