#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_matmul_standard {
    ($float:ident) => {
        type EG = $float;
        type ES = half::f16;

        $crate::matmul_standard_tests!();
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_standard {
            use super::*;
            ::paste::paste! {
                mod accelerated {
                    use super::*;
                    type TMM = $crate::matmul::components::tile::accelerated::Accelerated;

                    $(mod [<$float _ty>] {
                        use super::*;
                        $crate::testgen_matmul_standard!($float);
                    })*
                }
                mod plane {
                    use super::*;
                    type TMM = $crate::matmul::components::tile::plane::PlaneMma;

                    $(mod [<$float _ty>] {
                        use super::*;
                        $crate::testgen_matmul_standard!($float);
                    })*
                }

            }
        }
    };
}
