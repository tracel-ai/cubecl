use cubecl_core::prelude::*;
use cubecl_core::tensor_line_size_parallel;
use cubecl_core::{CubeElement, server};

use crate::matmul::components::Ident;
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::MatmulLaunch;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::MatmulSelection;
use crate::matmul::components::MatrixLayout;
use crate::matmul::components::global::args::TensorInputsLaunch;
use crate::matmul::kernels::matmul::Algorithm;
use crate::matmul::tests::test_utils::Sample;
use crate::matmul::tests::test_utils::TestPrecision;

#[derive(Debug)]
pub(crate) struct TensorRawParts<N: Numeric + CubeElement> {
    pub handle: server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub original_data: Option<Vec<N>>,
}

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_algorithm<A, P, R>(
    client: ComputeClient<R::Server, R::Channel>,
    mut problem: MatmulProblem,
    input: <A::BatchMatmul as MatmulConfigFactory>::Input,
    selection: MatmulSelection,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let env = std::env::var("MATMUL_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => match val.as_str() {
            "panic" => true,
            "skip" => false,
            _ => false,
        },
        Err(_) => false,
    };
    let lhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<P, R>(&client, &problem, Ident::Out);

    problem.lhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&P::EG::as_elem_native_unchecked()),
        &lhs.shape,
        &lhs.strides,
        match problem.lhs_layout {
            MatrixLayout::RowMajor => lhs.strides.len() - 1,
            MatrixLayout::ColMajor => lhs.strides.len() - 2,
        },
    );
    problem.rhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&P::EG::as_elem_native_unchecked()),
        &rhs.shape,
        &rhs.strides,
        match problem.rhs_layout {
            MatrixLayout::RowMajor => lhs.strides.len() - 1,
            MatrixLayout::ColMajor => lhs.strides.len() - 2,
        },
    );
    problem.out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&P::EG::as_elem_native_unchecked()),
        &out.shape,
        &out.strides,
        out.strides.len() - 1,
    );

    let cube_dim = A::cube_dim(&selection);
    let cube_count = A::cube_count(&selection, &problem);

    let config = match A::make_config(input, &problem, &cube_dim, &cube_count, P::QUANTIZED) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {err:?}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return;
            }
        }
    };

    if let Err(err) = A::check_availability::<R, (P::EG, P::ES, f32, P::EG)>(&client, &config) {
        let msg = format!("Skipped - not supported: {err:?}");
        if panic_on_launch_err {
            panic!("{msg}")
        } else {
            println!("{msg}");
            client.flush();
            return;
        }
    }

    unsafe {
        A::BatchMatmul::launch_unchecked::<(P::EG, P::ES, P::EA, P::EG), R>(
            &client,
            cube_dim,
            cube_count,
            TensorInputsLaunch::new(
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &lhs.handle,
                    &lhs.strides,
                    &lhs.shape,
                    problem.lhs_line_size,
                ),
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &rhs.handle,
                    &rhs.strides,
                    &rhs.shape,
                    problem.rhs_line_size,
                ),
            ),
            TensorArg::<R>::from_raw_parts::<P::EG>(
                &out.handle,
                &out.strides,
                &out.shape,
                problem.out_line_size,
            ),
            ScalarArg::new(problem.k as u32),
            config,
        );
    }

    P::assert_result::<R>(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );
}

fn tensor_raw_parts<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    ident: Ident,
) -> TensorRawParts<P::EG> {
    match ident {
        Ident::Lhs => {
            let mut original_data = P::EG::sample(tensor_size(problem, Ident::Lhs), 1234);

            if let Some(params) = P::quantization_params(Ident::Lhs) {
                original_data.extend_from_slice(&params.scaling);
                let zero = P::EG::from_int(0);
                original_data.extend_from_slice(&[zero, zero, zero, params.zero_offset]);
            }

            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };

            let handle = client.create(P::EG::as_bytes(&data));

            TensorRawParts {
                handle,
                shape: shape(problem, Ident::Lhs),
                strides: strides(problem, Ident::Lhs),
                original_data: Some(original_data),
            }
        }
        Ident::Rhs => {
            let mut original_data = P::EG::sample(tensor_size(problem, Ident::Rhs), 5678);

            if let Some(params) = P::quantization_params(Ident::Rhs) {
                original_data.extend_from_slice(&params.scaling);
                let zero = P::EG::from_int(0);
                original_data.extend_from_slice(&[zero, zero, zero, params.zero_offset]);
            }

            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            let handle = client.create(P::EG::as_bytes(&data));

            TensorRawParts {
                handle,
                shape: shape(problem, Ident::Rhs),
                strides: strides(problem, Ident::Rhs),
                original_data: Some(original_data),
            }
        }
        Ident::Out => {
            let zero = P::EG::from_int(0);

            let mut data = vec![zero; tensor_size(problem, Ident::Out)];

            if let Some(params) = P::quantization_params(Ident::Out) {
                data.extend_from_slice(&params.scaling);
                data.extend_from_slice(&[zero, zero, zero, params.zero_offset]);
            }

            let shape = shape(problem, Ident::Out);
            let (handle, strides) =
                client.create_tensor(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());
            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: None,
            }
        }
    }
}

pub(crate) fn transpose<E: Copy>(array: &[E], batches: usize, rows: usize, cols: usize) -> Vec<E> {
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

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &MatmulProblem, ident: Ident) -> usize {
    match ident {
        Ident::Lhs => problem.num_batches() * problem.m * problem.k,
        Ident::Rhs => problem.num_batches() * problem.k * problem.n,
        Ident::Out => problem.num_batches() * problem.m * problem.n,
    }
}

/// Returns the shape of the identified tensor, inferred by the problem definition
pub(crate) fn shape(problem: &MatmulProblem, ident: Ident) -> Vec<usize> {
    match ident {
        Ident::Lhs => problem
            .batches
            .0
            .iter()
            .cloned()
            .chain(vec![problem.m, problem.k])
            .collect(),
        Ident::Rhs => problem
            .batches
            .1
            .iter()
            .cloned()
            .chain(vec![problem.k, problem.n])
            .collect(),
        Ident::Out => problem
            .batch_dims()
            .iter()
            .cloned()
            .chain(vec![problem.m, problem.n])
            .collect(),
    }
}

/// Returns the stride of the identified tensor, inferred by the problem definition
pub(crate) fn strides(problem: &MatmulProblem, ident: Ident) -> Vec<usize> {
    let shape = shape(problem, ident);
    let rank = shape.len();
    let mut strides = Vec::with_capacity(rank);

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

    if rank > 2 {
        strides.push(last_batch);

        for b in shape.iter().rev().take(rank - 3) {
            strides.push(last_batch * b)
        }
    }

    strides.into_iter().rev().collect()
}
