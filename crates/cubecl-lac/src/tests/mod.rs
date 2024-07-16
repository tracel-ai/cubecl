#![allow(missing_docs)]

pub mod matmul_cube;

#[macro_export]
macro_rules! testgen_all {
    () => {
        mod lac {
            use super::*;

            cubecl_lac::testgen_matmul_cube!();
        }
    };
}
