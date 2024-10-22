use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matmul_batch::BatchMatmul;
use crate::matmul::matmul_batch::BmmConfig;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::problem::MatmulProblem;

use super::test_utils::assert_equals_approx;
use super::test_utils::generate_random_data;
use super::test_utils::matmul_cpu_reference;

pub fn test_matmul<MM, EG, B, G, R>(
    problem: MatmulProblem,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    config: MM::Config,
    device: &R::Device,
) where
    EG: Numeric + CubeElement,
    MM: BatchMatmul<EG, B>,
    B: BmmConfig,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    assert!(problem.m <= config.max_m() && problem.n <= config.max_n());

    // Lhs
    assert!(match problem.lhs_layout {
        MatrixLayout::RowMajor => problem.k % problem.lhs_line_size as u32 == 0,
        MatrixLayout::ColMajor => problem.m % problem.lhs_line_size as u32 == 0,
    });

    // Rhs
    assert!(match problem.rhs_layout {
        MatrixLayout::RowMajor => problem.n % problem.rhs_line_size as u32 == 0,
        MatrixLayout::ColMajor => problem.k % problem.rhs_line_size as u32 == 0,
    });

    // Out
    assert!(problem.n % problem.out_line_size as u32 == 0);

    let lhs_size = problem.m * problem.k;
    let rhs_size = problem.k * problem.n;
    let out_size = problem.m * problem.n;

    let lhs_original_data: Vec<f32> = generate_random_data(lhs_size as usize);
    let rhs_original_data: Vec<f32> = generate_random_data(rhs_size as usize);

    let (lhs_data, lhs_strides) = match problem.lhs_layout {
        MatrixLayout::RowMajor => (lhs_original_data.clone(), [problem.k as usize, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&lhs_original_data, problem.m as usize, problem.k as usize),
            [1, problem.m as usize],
        ),
    };
    let (rhs_data, rhs_strides) = match problem.rhs_layout {
        MatrixLayout::RowMajor => (rhs_original_data.clone(), [problem.n as usize, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&rhs_original_data, problem.k as usize, problem.n as usize),
            [1, problem.k as usize],
        ),
    };

    let lhs = client.create(EG::as_bytes(&EG::from_values(&lhs_data)));
    let rhs = client.create(EG::as_bytes(&EG::from_values(&rhs_data)));
    let out = client.empty(out_size as usize * EG::as_elem().size());

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(
                &lhs,
                &lhs_strides,
                &[problem.m as usize, problem.k as usize],
                problem.lhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                &rhs,
                &rhs_strides,
                &[problem.k as usize, problem.n as usize],
                problem.rhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                &out,
                &[problem.n as usize, 1],
                &[problem.m as usize, problem.n as usize],
                problem.out_line_size,
            ),
            config,
        );
    }

    let expected = matmul_cpu_reference(&lhs_original_data, &rhs_original_data, problem);
    if let Err(e) = assert_equals_approx::<EG, R>(&client, out, &expected, 10e-2) {
        panic!("{}", e);
    }
}

fn transpose<E: Copy>(array: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut result = vec![array[0]; array.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = array[i * cols + j];
        }
    }
    result
}
