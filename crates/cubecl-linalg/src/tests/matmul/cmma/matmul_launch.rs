#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_launch {
    () => {
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul_launch;
        use cubecl_linalg::tensor::TensorHandle;

        #[test]
        pub fn test_launch_matmul_b3x4_g300x200x250_col_row() {
            let client: ComputeClient<
                <TestRuntime as Runtime>::Server,
                <TestRuntime as Runtime>::Channel,
            > = TestRuntime::client(&Default::default());

            let problem = MatmulProblem {
                m: 300,
                n: 200,
                k: 250,
                b: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
                _element: PhantomData,
            };

            test_matmul_launch::<f32, TestRuntime>(&client, problem);
        }
    };
}
