use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix::MatrixLayout;
use crate::matmul::problem::MatmulProblem;
use crate::matmul::Matmul;

use super::test_utils::assert_equals_approx;
use super::test_utils::generate_random_data;
use super::test_utils::matmul_cpu_reference;

pub fn test_matmul<MM, I, O, R>(problem: MatmulProblem, num_planes: u32, device: &R::Device)
where
    I: Numeric + CubeElement,
    O: Numeric + CubeElement,
    MM: Matmul<I, O>,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
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

    let lhs = client.create(I::as_bytes(&I::from_values(&lhs_data)));
    let rhs = client.create(I::as_bytes(&I::from_values(&rhs_data)));
    let out = client.empty(out_size as usize * O::as_elem().size());

    let cube_dim = CubeDim::new(32, num_planes, 1);
    let cube_count = CubeCount::Static(1, 1, 1);
    let config = MM::Config::default(cube_dim, cube_count, problem);

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
    if let Err(e) = assert_equals_approx::<I, R>(&client, out, &expected, 10e-2) {
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
