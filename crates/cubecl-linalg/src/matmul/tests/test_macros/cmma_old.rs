#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_cmma_old {
    () => {
        use super::*;

        #[test]
        pub fn test_matmul_cmma_old_all() {
            cubecl_linalg::matmul::tests::cmma_old::table_test::test_cmma_all::<TestRuntime>(
                &Default::default(),
            )
        }
    };
}
