use cubecl_core::{
    CubeElement,
    server::{self, Allocation},
};
use cubecl_core::{prelude::*, server::AllocationDescriptor};

use crate::components::MatrixLayout;
use crate::components::batch::{BatchConfig, BatchMatmulFamily};
use crate::components::global::args::TensorInputsLaunch;
use crate::components::{AvailableLineSizes, MatmulIdent};
use crate::components::{MatmulProblem, MatmulSelection};
use crate::kernels::layered::Algorithm;
use crate::tests::test_utils::Sample;
use crate::tests::test_utils::TestPrecision;

#[derive(Debug)]
pub struct TensorRawParts<N: Numeric + CubeElement> {
    pub handle: server::Handle,
    pub scale: Option<server::Handle>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub original_data: Option<Vec<N>>,
}

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_algorithm<A, P, R>(
    client: ComputeClient<R::Server, R::Channel>,
    problem: MatmulProblem,
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
    let lhs = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Out);

    let line_sizes = AvailableLineSizes::from_elem_types::<R>(
        &P::EG::as_elem_native_unchecked(),
        &P::EG::as_elem_native_unchecked(),
        &P::EG::as_elem_native_unchecked(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs_with_tensor(&lhs.strides, &lhs.shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&rhs.strides, &rhs.shape, problem.rhs_layout)
        .filter_out_with_tensor(&out.strides, &out.shape)
        .pick_max()
        .unwrap();

    let config = match A::setup::<(P::EG, P::EG, P::ES, P::ES, P::EA, P::EG), R>(
        &client,
        &problem,
        &selection,
        &line_sizes,
    ) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {err}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return;
            }
        }
    };

    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    unsafe {
        A::BatchMatmul::launch_unchecked::<P::MP, R>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &lhs.handle,
                    &lhs.strides,
                    &lhs.shape,
                    line_sizes.lhs,
                ),
                lhs.scale
                    .as_ref()
                    .map(|it| TensorArg::<R>::from_raw_parts::<P::EG>(it, &[1], &[1], 1))
                    .into(),
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &rhs.handle,
                    &rhs.strides,
                    &rhs.shape,
                    line_sizes.rhs,
                ),
                rhs.scale
                    .as_ref()
                    .map(|it| TensorArg::<R>::from_raw_parts::<P::EG>(it, &[1], &[1], 1))
                    .into(),
            ),
            TensorArg::<R>::from_raw_parts::<P::EG>(
                &out.handle,
                &out.strides,
                &out.shape,
                line_sizes.out,
            ),
            cube_count_plan.as_args(),
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
    ident: MatmulIdent,
) -> TensorRawParts<P::EG> {
    match ident {
        MatmulIdent::Lhs => {
            let mut tensor_shape = problem.shape(MatmulIdent::Lhs);

            let handle = P::EG::sample::<R>(client, &tensor_shape, 1234);

            let data = client.read_one(handle.handle);
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = tensor_shape.len();

            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    tensor_shape.swap(rank - 1, rank - 2);
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };
            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), size_of::<P::EG>()),
                P::EG::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors(descriptors);
            let Allocation {
                handle,
                mut strides,
            } = tensors.remove(0);

            if matches!(problem.lhs_layout, MatrixLayout::ColMajor) {
                tensor_shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Rhs => {
            let mut tensor_shape = problem.shape(MatmulIdent::Rhs);

            let handle = P::EG::sample::<R>(client, &tensor_shape, 5678);

            let data = client.read_one(handle.handle);
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = tensor_shape.len();

            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    tensor_shape.swap(rank - 1, rank - 2);
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), size_of::<P::EG>()),
                P::EG::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors(descriptors);
            let Allocation {
                handle,
                mut strides,
            } = tensors.remove(0);
            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            if matches!(problem.rhs_layout, MatrixLayout::ColMajor) {
                tensor_shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let tensor_shape = problem.shape(MatmulIdent::Out);

            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), size_of::<P::EG>()),
                P::EG::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors(descriptors);
            let Allocation { handle, strides } = tensors.remove(0);
            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
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
pub(crate) fn tensor_size(problem: &MatmulProblem, ident: MatmulIdent) -> usize {
    match ident {
        MatmulIdent::Lhs => problem.num_batches() * problem.m * problem.k,
        MatmulIdent::Rhs => problem.num_batches() * problem.k * problem.n,
        MatmulIdent::Out => problem.num_batches() * problem.m * problem.n,
    }
}

/// Returns the stride of the identified tensor, inferred by the problem definition
pub(crate) fn strides(problem: &MatmulProblem, ident: MatmulIdent) -> Vec<usize> {
    let shape = problem.shape(ident);
    let rank = shape.len();
    let mut strides = Vec::with_capacity(rank);

    let (last_batch, x, y) = match ident {
        MatmulIdent::Lhs => match problem.lhs_layout {
            MatrixLayout::RowMajor => (problem.m * problem.k, problem.k, 1),
            MatrixLayout::ColMajor => (problem.m * problem.k, 1, problem.m),
        },
        MatmulIdent::Rhs => match problem.rhs_layout {
            MatrixLayout::RowMajor => (problem.k * problem.n, problem.n, 1),
            MatrixLayout::ColMajor => (problem.k * problem.n, 1, problem.k),
        },
        MatmulIdent::Out => (problem.m * problem.n, problem.n, 1),
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
