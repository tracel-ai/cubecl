use std::fmt::Display;

use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_core::CubeElement;
use cubecl_core::Feature;

use crate::matmul::components::batch;
use crate::matmul::components::Ident;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::MatrixLayout;
use crate::matmul::kernels::cmma_matmul;
use crate::matmul::tests::test_utils::CastInto;
use crate::tensor::TensorHandle;

use crate::matmul::tests::test_utils::assert_equals_approx;
use crate::matmul::tests::test_utils::generate_random_data;
use crate::matmul::tests::test_utils::matmul_cpu_reference;

struct TensorRawParts<F: Float + CubeElement> {
    handle: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    original_data: Option<Vec<F>>,
}

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_internal<MM, EG, ES, B, R>(
    problem: MatmulProblem<EG>,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    config: MM::Config,
    device: &R::Device,
) where
    EG: Float + CubeElement + Display + CastInto<ES>,
    ES: Float + CubeElement + Display + CastInto<EG>,
    MM: batch::Matmul<EG>,
    B: batch::Config,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    if !(client.properties().feature_enabled(Feature::Subcube)
        && client
            .properties()
            .feature_enabled(Feature::Type(EG::as_elem())))
    {
        // Can't execute the test.
        return;
    }

    let lhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Out);

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts::<EG>(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                problem.lhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts::<EG>(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                problem.rhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts::<EG>(
                &out.handle,
                &out.strides,
                &out.shape,
                problem.out_line_size,
            ),
            config,
        );
    }

    assert_result::<EG, ES, R>(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        None,
    );
}

/// Test the correctness of the high-level Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_launch<EG: Float + CubeElement + Display + CastInto<EG>, R: Runtime>(
    problem: MatmulProblem<EG>,
    disable_cmma: bool,
    device: &R::Device,
) {
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    if !(client.properties().feature_enabled(Feature::Subcube)
        && client
            .properties()
            .feature_enabled(Feature::Type(EG::as_elem())))
    {
        // Can't execute the test.
        return;
    }

    let lhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<EG, R>(&client, &problem, Ident::Out);

    let out = cmma_matmul::launch::<R, EG>(
        &client,
        TensorHandle::new(lhs.shape, lhs.strides, lhs.handle),
        TensorHandle::new(rhs.shape, rhs.strides, rhs.handle),
        TensorHandle::new(out.shape, out.strides, out.handle),
        disable_cmma,
    );

    assert_result::<EG, EG, R>(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        // We cannot assume the inner precision of the matmul, therefore we need a permissive epsilon
        Some(10e-2),
    );
}

fn tensor_raw_parts<EG: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem<EG>,
    ident: Ident,
) -> TensorRawParts<EG> {
    match ident {
        Ident::Lhs => {
            let original_data: Vec<EG> = generate_random_data(tensor_size(problem, Ident::Lhs));
            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    transpose::<EG>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };

            TensorRawParts {
                handle: client.create(EG::as_bytes(&data)),
                shape: shape(problem, Ident::Lhs),
                strides: strides(problem, Ident::Lhs),
                original_data: Some(original_data),
            }
        }
        Ident::Rhs => {
            let original_data: Vec<EG> = generate_random_data(tensor_size(problem, Ident::Rhs));
            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    transpose::<EG>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            TensorRawParts {
                handle: client.create(EG::as_bytes(&data)),
                shape: shape(problem, Ident::Rhs),
                strides: strides(problem, Ident::Rhs),
                original_data: Some(original_data),
            }
        }
        Ident::Out => {
            let handle = client.empty(tensor_size(problem, Ident::Out) * EG::as_elem().size());
            let shape = shape(problem, Ident::Out);
            let strides = strides(problem, Ident::Out);

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

fn assert_result<
    EG: Float + CubeElement + Display + CastInto<ES>,
    ES: Float + CubeElement + CastInto<EG>,
    R: Runtime,
>(
    lhs: &[EG],
    rhs: &[EG],
    problem: &MatmulProblem<EG>,
    client: &ComputeClient<R::Server, R::Channel>,
    out: Handle,
    epsilon: Option<f32>,
) {
    let epsilon = match epsilon {
        Some(epsilon) => epsilon,
        None => {
            let maybe_cmma = client.properties().feature_enabled(Feature::Cmma {
                a: ES::as_elem(),
                b: ES::as_elem(),
                c: EG::as_elem(),
                m: 16,
                k: 16,
                n: 16,
            });

            // Need to compensate for the temporary conversion to f16/tf32
            match maybe_cmma {
                true => 10e-5 / EG::EPSILON.to_f32().unwrap() * half::f16::EPSILON.to_f32(),
                false => 10e-5,
            }
        }
    };

    let expected = matmul_cpu_reference(lhs, rhs, problem);
    if let Err(e) = assert_equals_approx::<R, EG>(client, out, &expected, epsilon) {
        panic!("{}", e);
    }
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
fn tensor_size<EG: Numeric>(problem: &MatmulProblem<EG>, ident: Ident) -> usize {
    match ident {
        Ident::Lhs => problem.num_batches() * problem.m * problem.k,
        Ident::Rhs => problem.num_batches() * problem.k * problem.n,
        Ident::Out => problem.num_batches() * problem.m * problem.n,
    }
}

/// Returns the shape of the identified tensor, inferred by the problem definition
fn shape<EG: Numeric>(problem: &MatmulProblem<EG>, ident: Ident) -> Vec<usize> {
    problem
        .batches
        .iter()
        .cloned()
        .chain(match ident {
            Ident::Lhs => vec![problem.m, problem.k],
            Ident::Rhs => vec![problem.k, problem.n],
            Ident::Out => vec![problem.m, problem.n],
        })
        .collect()
}

/// Returns the stride of the identified tensor, inferred by the problem definition
pub(crate) fn strides<EG: Numeric>(problem: &MatmulProblem<EG>, ident: Ident) -> Vec<usize> {
    let mut strides = Vec::with_capacity(problem.batches.len() + 2);

    let (last_batch, x, y) = match ident {
        Ident::Lhs => match problem.lhs_layout {
            MatrixLayout::RowMajor => (problem.m * problem.k, problem.k, 1),
            MatrixLayout::ColMajor => (problem.m * problem.k, 1, problem.m),
        },
        Ident::Rhs => match problem.rhs_layout {
            MatrixLayout::RowMajor => (problem.k * problem.n, problem.n, 1),
            MatrixLayout::ColMajor => (problem.k * problem.n, 1, problem.k),
        },
        Ident::Out => (problem.m * problem.n, problem.n, 1),
    };

    strides.push(y);
    strides.push(x);

    if !problem.batches.is_empty() {
        strides.push(last_batch);

        for b in problem.batches.iter().rev().take(problem.batches.len() - 1) {
            strides.push(last_batch * b)
        }
    }

    strides.into_iter().rev().collect()
}
