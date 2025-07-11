use cubecl_core::prelude::*;
use cubecl_core::{CubeElement, server};

use crate::components::args::TensorInputsLaunch;
use crate::components::{AttentionProblem, AttentionSelection};
use crate::components::{AvailableLineSizes, Ident};
use crate::kernels::Algorithm;
use crate::tests::test_utils::Sample;
use crate::tests::test_utils::TestPrecision;

#[derive(Debug)]
pub struct TensorRawParts<N: Numeric + CubeElement> {
    pub handle: server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub original_data: Option<Vec<N>>,
}

/// Test the correctness of the specified Attention on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_attention_algorithm<A, P, R>(
    client: ComputeClient<R::Server, R::Channel>,
    problem: AttentionProblem,
    selection: AttentionSelection,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let env = std::env::var("ATTENTION_TEST_MODE");

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

    let line_sizes = AvailableLineSizes::from_elem_types::<R>(
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

    let config = match A::setup::<(P::EG, P::ES, P::EA, P::EG), R>(
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
        lhs.quant_params,
        &rhs.original_data.unwrap(),
        rhs.quant_params,
        &problem,
        &client,
        out.handle,
        out.quant_params,
        &out.shape,
        &out.strides,
    );
}

fn tensor_raw_parts<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &AttentionProblem,
    ident: Ident,
) -> TensorRawParts<P::EG> {
    match ident {
        Ident::Query => {
            let tensor_shape = shape(problem, Ident::Query);

            let handle = P::EG::sample::<R>(client, &tensor_shape, 123);

            let data = client.read_one(handle.handle.binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let data = original_data.clone();
            let data = P::EG::as_bytes(&data);
            let shape = tensor_shape.as_slice();
            let elem_size = size_of::<P::EG>();
            let (handle, strides) = client.create_tensor(data, shape, elem_size);

            TensorRawParts {
                handle,
                shape: tensor_shape.to_vec(),
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Key => {
            let tensor_shape = shape(problem, Ident::Key);

            let handle = P::EG::sample::<R>(client, &tensor_shape, 456);

            let data = client.read_one(handle.handle.binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let data = original_data.clone();
            let data = P::EG::as_bytes(&data);
            let shape = tensor_shape.as_slice();
            let elem_size = size_of::<P::EG>();
            let (handle, strides) = client.create_tensor(data, shape, elem_size);

            TensorRawParts {
                handle,
                shape: tensor_shape.to_vec(),
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Value => {
            let tensor_shape = shape(problem, Ident::Value);

            let handle = P::EG::sample::<R>(client, &tensor_shape, 789);

            let data = client.read_one(handle.handle.binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let data = original_data.clone();
            let data = P::EG::as_bytes(&data);
            let shape = tensor_shape.as_slice();
            let elem_size = size_of::<P::EG>();
            let (handle, strides) = client.create_tensor(data, shape, elem_size);

            TensorRawParts {
                handle,
                shape: tensor_shape.to_vec(),
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Mask => {
            let tensor_shape = shape(problem, Ident::Mask);

            let handle = P::EM::sample::<R>(client, &tensor_shape, 159);

            let data = client.read_one(handle.handle.binding());
            let data = P::EM::from_bytes(&data);
            let original_data = data.to_owned();

            let data = original_data.clone();
            let data = P::EM::as_bytes(&data);
            let shape = tensor_shape.as_slice();
            let elem_size = size_of::<P::EG>();
            let (handle, strides) = client.create_tensor(data, shape, elem_size);

            TensorRawParts {
                handle,
                shape: tensor_shape.to_vec(),
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, Ident::Out)];

            let tensor_shape = shape(problem, Ident::Out);

            let data = P::EG::as_bytes(&data);
            let shape = tensor_shape.as_slice();
            let elem_size = size_of::<P::EG>();
            let (handle, strides) = client.create_tensor(data, shape, elem_size);

            TensorRawParts {
                handle,
                shape: tensor_shape.to_vec(),
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
pub(crate) fn tensor_size(problem: &AttentionProblem, ident: Ident) -> usize {
    shape(problem, ident).iter().product()
}

pub(crate) fn shape(problem: &AttentionProblem, ident: Ident) -> [usize; 4] {
    match ident {
        Ident::Query => [
            problem.batch,
            problem.seq_q,
            problem.num_heads,
            problem.head_dim,
        ],
        Ident::Key => [
            problem.batch,
            problem.seq_k,
            problem.num_heads,
            problem.head_dim,
        ],
        Ident::Value => [
            problem.batch,
            problem.seq_k,
            problem.num_heads,
            problem.head_dim,
        ],
        Ident::Mask => [
            problem.batch,
            problem.seq_q,
            problem.num_heads,
            problem.seq_k,
        ],
        Ident::Out => [
            problem.batch,
            problem.seq_q,
            problem.num_heads,
            problem.head_dim,
        ],
    }
}

pub(crate) fn strides(problem: &AttentionProblem, ident: Ident) -> Vec<usize> {
    let shape = shape(problem, ident);

    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }

    strides
}
