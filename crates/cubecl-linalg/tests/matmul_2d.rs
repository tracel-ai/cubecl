use cubecl_core::{Runtime, client::ComputeClient, prelude::*};
use cubecl_linalg::matmul::{
    components::{MatmulPrecision, MatmulProblem},
    kernels::matmul::launch_ref,
};

#[test]
fn test_2d_input_matmul() {
    let client = ComputeClient::new().unwrap();

    // Test case 1: 1xN input
    let m = 1;
    let n = 1024;
    let k = 512;

    let problem = MatmulProblem {
        m,
        n,
        k,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        out_layout: MatrixLayout::RowMajor,
        batches: (1, 1),
        lhs_line_size: n,
        rhs_line_size: k,
        out_line_size: n,
    };

    // Create input tensors
    let lhs = client.create_tensor::<f32>(&[1, m, k]).unwrap();
    let rhs = client.create_tensor::<f32>(&[1, k, n]).unwrap();
    let out = client.create_tensor::<f32>(&[1, m, n]).unwrap();

    // Launch matmul
    let result = launch_ref::<_, f32>(&Default::default(), &client, &lhs, &rhs, &out);
    assert!(result.is_ok(), "Matmul should succeed for 1xN input");

    // Test case 2: Mx1 input
    let m = 1024;
    let n = 1;
    let k = 512;

    let problem = MatmulProblem {
        m,
        n,
        k,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        out_layout: MatrixLayout::RowMajor,
        batches: (1, 1),
        lhs_line_size: k,
        rhs_line_size: n,
        out_line_size: n,
    };

    // Create input tensors
    let lhs = client.create_tensor::<f32>(&[1, m, k]).unwrap();
    let rhs = client.create_tensor::<f32>(&[1, k, n]).unwrap();
    let out = client.create_tensor::<f32>(&[1, m, n]).unwrap();

    // Launch matmul
    let result = launch_ref::<_, f32>(&Default::default(), &client, &lhs, &rhs, &out);
    assert!(result.is_ok(), "Matmul should succeed for Mx1 input");
}
