use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_core::CubeElement;

use crate::matmul;
use crate::matmul::matmul_batch::BatchMatmul;
use crate::matmul::matmul_batch::BmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::problem::MatmulProblem;
use crate::tensor::TensorHandle;

use super::test_utils::assert_equals_approx;
use super::test_utils::generate_random_data;
use super::test_utils::matmul_cpu_reference;

struct TensorRawParts {
    handle: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    original_data: Option<Vec<f32>>,
}

pub fn test_matmul_internal<MM, EG, B, G, R>(
    problem: MatmulProblem<EG>,
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

    let lhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Out);

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                problem.lhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                problem.rhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                &out.handle,
                &out.strides,
                &out.shape,
                problem.out_line_size,
            ),
            config,
        );
    }

    assert_result::<EG, R>(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
    );
}

pub fn test_matmul_launch<EG: Numeric + CubeElement, R: Runtime>(
    problem: MatmulProblem<EG>,
    device: &R::Device,
) {
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    let lhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Out);

    let out = matmul::launch::<R, EG>(
        &client,
        TensorHandle::new(lhs.shape, lhs.strides, lhs.handle),
        TensorHandle::new(rhs.shape, rhs.strides, rhs.handle),
        TensorHandle::new(out.shape, out.strides, out.handle),
    );

    assert_result::<EG, R>(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
    );
}

fn tensor_raw_parts<EG: Numeric + CubeElement, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem<EG>,
    ident: Ident,
) -> TensorRawParts {
    match ident {
        Ident::Lhs => {
            let original_data: Vec<f32> =
                generate_random_data(problem.tensor_size(Ident::Lhs) as usize);
            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => transpose::<f32>(
                    &original_data,
                    problem.num_batches() as usize,
                    problem.m as usize,
                    problem.k as usize,
                ),
            };

            let handle = client.create(EG::as_bytes(&EG::from_values(&data)));
            let shape = problem.shape(Ident::Lhs);
            let strides = problem.strides(Ident::Lhs);

            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Rhs => {
            let original_data: Vec<f32> =
                generate_random_data(problem.tensor_size(Ident::Rhs) as usize);
            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => transpose::<f32>(
                    &original_data,
                    problem.num_batches() as usize,
                    problem.k as usize,
                    problem.n as usize,
                ),
            };
            let handle = client.create(EG::as_bytes(&EG::from_values(&data)));
            let shape = problem.shape(Ident::Rhs);
            let strides = problem.strides(Ident::Rhs);

            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Out => {
            let handle =
                client.empty(problem.tensor_size(Ident::Out) as usize * EG::as_elem().size());
            let shape = problem.shape(Ident::Out);
            let strides = problem.strides(Ident::Out);

            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: None,
            }
        }
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

fn assert_result<EG: Numeric + CubeElement, R: Runtime>(
    lhs: &Vec<f32>,
    rhs: &Vec<f32>,
    problem: &MatmulProblem<EG>,
    client: &ComputeClient<R::Server, R::Channel>,
    out: Handle,
) {
    let expected = matmul_cpu_reference(&lhs, &rhs, problem);
    if let Err(e) = assert_equals_approx::<EG, R>(&client, out, &expected, 10e-2) {
        panic!("{}", e);
    }
}
