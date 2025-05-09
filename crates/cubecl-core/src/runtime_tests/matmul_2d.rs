use cubecl_core::{Runtime, client::ComputeClient, prelude::*};
use cubecl_linalg::matmul::{
    components::{MatrixLayout, MatmulProblem},
    kernels::matmul::base::matmul_cmma_ref,
};

#[test]
fn test_2d_input_matmul() {
    let client = ComputeClient::new().unwrap();
    
    // Test 1xN input
    let m = 1;
    let n = 32;
    let k = 16;
    
    let lhs = client.create_tensor::<f32>(&[m, k]).unwrap();
    let rhs = client.create_tensor::<f32>(&[k, n]).unwrap();
    let out = client.create_tensor::<f32>(&[m, n]).unwrap();
    
    let problem = MatmulProblem {
        m,
        n,
        k,
        batches: (vec![1], vec![1]),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        out_layout: MatrixLayout::RowMajor,
        lhs_line_size: 16,
        rhs_line_size: 16,
        out_line_size: 16,
    };
    
    matmul_cmma_ref::<_, f32, _>(&client, &lhs, &rhs, &out, (false, false)).unwrap();
    
    // Test Mx1 input
    let m = 32;
    let n = 1;
    let k = 16;
    
    let lhs = client.create_tensor::<f32>(&[m, k]).unwrap();
    let rhs = client.create_tensor::<f32>(&[k, n]).unwrap();
    let out = client.create_tensor::<f32>(&[m, n]).unwrap();
    
    let problem = MatmulProblem {
        m,
        n,
        k,
        batches: (vec![1], vec![1]),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        out_layout: MatrixLayout::RowMajor,
        lhs_line_size: 16,
        rhs_line_size: 16,
        out_line_size: 16,
    };
    
    matmul_cmma_ref::<_, f32, _>(&client, &lhs, &rhs, &out, (false, false)).unwrap();
} 