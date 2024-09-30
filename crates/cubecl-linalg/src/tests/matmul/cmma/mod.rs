#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_instruction;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_instruction!();
        cubecl_linalg::testgen_cmma_matmul!();
    };
}
