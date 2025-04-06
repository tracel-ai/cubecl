#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_conv2d_accelerated {
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        $crate::conv2d_standard_tests!();
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated::Accelerated;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_conv2d_accelerated!($float, $float);
                })*
            }
        }
    };
}
