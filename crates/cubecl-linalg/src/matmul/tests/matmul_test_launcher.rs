use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matmul_batch::BatchMatmul;
use crate::matmul::matmul_batch::BmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::problem::MatmulProblem;

use super::test_utils::assert_equals_approx;
use super::test_utils::generate_random_data;
use super::test_utils::matmul_cpu_reference;

pub fn test_matmul_internal<MM, EG, B, G, R>(
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

    let lhs_shape = problem.shape(Ident::Lhs);
    let rhs_shape = problem.shape(Ident::Rhs);
    let out_shape = problem.shape(Ident::Out);
    let lhs_strides = problem.strides(Ident::Lhs);
    let rhs_strides = problem.strides(Ident::Rhs);
    let out_strides = problem.strides(Ident::Out);

    let lhs_original_data: Vec<f32> =
        generate_random_data(problem.tensor_size(Ident::Lhs) as usize);
    let lhs_data = match problem.lhs_layout {
        MatrixLayout::RowMajor => lhs_original_data.clone(),
        MatrixLayout::ColMajor => transpose::<f32>(
            &lhs_original_data,
            problem.num_batches() as usize,
            problem.m as usize,
            problem.k as usize,
        ),
    };
    let rhs_original_data: Vec<f32> =
        generate_random_data(problem.tensor_size(Ident::Rhs) as usize);
    let rhs_data = match problem.rhs_layout {
        MatrixLayout::RowMajor => rhs_original_data.clone(),
        MatrixLayout::ColMajor => transpose::<f32>(
            &rhs_original_data,
            problem.num_batches() as usize,
            problem.k as usize,
            problem.n as usize,
        ),
    };

    let lhs = client.create(EG::as_bytes(&EG::from_values(&lhs_data)));
    let rhs = client.create(EG::as_bytes(&EG::from_values(&rhs_data)));
    let out = client.empty(problem.tensor_size(Ident::Out) as usize * EG::as_elem().size());

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(&lhs, &lhs_strides, &lhs_shape, problem.lhs_line_size),
            TensorArg::<R>::from_raw_parts(&rhs, &rhs_strides, &rhs_shape, problem.rhs_line_size),
            TensorArg::<R>::from_raw_parts(&out, &out_strides, &out_shape, problem.out_line_size),
            config,
        );
    }

    let expected = matmul_cpu_reference(&lhs_original_data, &rhs_original_data, problem);
    if let Err(e) = assert_equals_approx::<EG, R>(&client, out, &expected, 10e-2) {
        panic!("{}", e);
    }
}

fn transpose<E: Copy>(array: &[E], batches: usize, rows: usize, cols: usize) -> Vec<E> {
    let mut result = vec![array[0]; array.len()];
    for b in 0..batches {
        for i in 0..rows {
            for j in 0..cols {
                result[(b * rows * cols) + j * rows + i] = array[(b * rows * cols) + i * cols + j];
            }
        }
    }
    result
}

pub fn test_matmul_launch<R: Runtime>(
    _client: &ComputeClient<R::Server, R::Channel>,
    _problem: MatmulProblem,
    // lhs: TensorHandle<R, E>,
    // rhs: TensorHandle<R, E>,
    // out: TensorHandle<R, E>,
) {
    todo!()
    // let lhs_original_data: Vec<f32> =
    //     generate_random_data(problem.tensor_size(Ident::Lhs) as usize);
    // let lhs_data = match problem.lhs_layout {
    //     MatrixLayout::RowMajor => lhs_original_data.clone(),
    //     MatrixLayout::ColMajor => transpose::<f32>(
    //         &lhs_original_data,
    //         problem.num_batches() as usize,
    //         problem.m as usize,
    //         problem.k as usize,
    //     ),
    // };
    // let rhs_original_data: Vec<f32> =
    //     generate_random_data(problem.tensor_size(Ident::Rhs) as usize);
    // let rhs_data = match problem.rhs_layout {
    //     MatrixLayout::RowMajor => rhs_original_data.clone(),
    //     MatrixLayout::ColMajor => transpose::<f32>(
    //         &rhs_original_data,
    //         problem.num_batches() as usize,
    //         problem.k as usize,
    //         problem.n as usize,
    //     ),
    // };

    // let lhs = client.create(EG::as_bytes(&EG::from_values(&lhs_data)));
    // let rhs = client.create(EG::as_bytes(&EG::from_values(&rhs_data)));
    // let out = client.empty(problem.tensor_size(Ident::Out) as usize * EG::as_elem().size());

    // matmul::launch(client, lhs, rhs, out)
}
