#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_launch {
    () => {
        use cubecl_linalg::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_launch;
        use cubecl_linalg::tensor::TensorHandle;

        #[test]
        pub fn test_launch_matmul_b3x4_g300x200x250_col_row() {
            type EG = f32;
            let problem = MatmulProblem::<EG> {
                m: 300,
                n: 200,
                k: 250,
                batches: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
                _element: PhantomData,
            };

            test_matmul_launch::<EG, TestRuntime>(problem, false, &Default::default());
        }
    };
}
