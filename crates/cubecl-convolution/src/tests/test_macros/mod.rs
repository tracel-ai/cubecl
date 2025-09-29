#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_conv2d_accelerated {
    ([$($float:ident: $acc: ident),*]) => {
        #[allow(non_snake_case)]
        mod conv2d_accelerated {
            use super::*;
            use cubecl_std::CubeOption;
            use cubecl_matmul::components::tile::io::Strided;
            type TMM = cubecl_matmul::components::tile::accelerated::AcceleratedMatmul<CubeOption<Strided>>;

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
