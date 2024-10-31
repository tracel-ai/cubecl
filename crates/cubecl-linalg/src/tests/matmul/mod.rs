#![allow(missing_docs)]

pub mod cmma;
pub mod tiling2d;

#[macro_export]
macro_rules! testgen_all {
    () => {
        mod linalg {
            $crate::testgen_all!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;

            pub type FloatT = $float;

            cubecl_linalg::testgen_cmma!();
            cubecl_linalg::testgen_tiling2d!();
    };
    ([$($float:ident),*]) => {
        mod linalg {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_all!($float);
                })*
            }
        }
    };
}
