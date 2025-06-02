#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_conv2d_accelerated {
    ([$($float:ident: $acc: ident),*]) => {
        #[allow(non_snake_case)]
        mod conv2d_accelerated {
            use super::*;
            type TMM = cubecl_matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_conv2d_accelerated!($float, $acc);
                })*
            }
        }
    };
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        #[cfg(feature="conv_tests")]
        $crate::conv2d_standard_tests!();
    };
}
