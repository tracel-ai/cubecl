use cubecl_core::prelude::*;
use cubecl_core::{CubeElement, server};

use crate::components::args::TensorInputsLaunch;
use crate::components::batch::BatchAttentionFamily;
use crate::components::batch::BatchAttentionConfig;
use crate::components::{AttentionProblem, AttentionSelection};
use crate::components::{AvailableLineSizes, Ident};
use crate::kernels::Algorithm;
use crate::tests::test_utils::Sampleable;
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
    let query = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, Ident::Query, 12);
    let key = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, Ident::Key, 34);
    let value = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, Ident::Value, 56);
    // let mask = tensor_raw_parts_input::<P, R, P::EM>(&client, &problem, Ident::Mask, 78);
    let out = tensor_raw_parts_output::<P, R>(&client, &problem);

    let line_sizes = AvailableLineSizes::from_elem_types::<R>(
        &P::EG::as_elem_native_unchecked(),
        &P::EM::as_elem_native_unchecked(),
        &P::EG::as_elem_native_unchecked(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_with_tensor(Ident::Query, &query.strides, &query.shape)
        .filter_with_tensor(Ident::Key, &key.strides, &key.shape)
        .filter_with_tensor(Ident::Value, &value.strides, &value.shape)
        // .filter_with_tensor(Ident::Mask, &mask.strides, &mask.shape)
        .filter_with_tensor(Ident::Out, &out.strides, &out.shape)
        .pick_max()
        .unwrap();

    let config = match A::setup::<(P::EG, P::EM, P::ES, P::EA, P::EG), R>(
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
        A::BatchAttention::launch_unchecked::<P::MP, R>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &query.handle,
                    &query.strides,
                    &query.shape,
                    line_sizes.query,
                ),
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &key.handle,
                    &key.strides,
                    &key.shape,
                    line_sizes.key,
                ),
                TensorArg::<R>::from_raw_parts::<P::EG>(
                    &value.handle,
                    &value.strides,
                    &value.shape,
                    line_sizes.value,
                ),
                // TensorArg::<R>::from_raw_parts::<P::EM>(
                //     &mask.handle,
                //     &mask.strides,
                //     &mask.shape,
                //     line_sizes.mask,
                // ),
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
        &query.original_data.unwrap(),
        &key.original_data.unwrap(),
        &value.original_data.unwrap(),
        None,
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );
}

fn tensor_raw_parts_input<P: TestPrecision, R: Runtime, T>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &AttentionProblem,
    ident: Ident,
    sample_seed: u64,
) -> TensorRawParts<T>
where
    T: Numeric + CubeElement + Sampleable,
{
    let tensor_shape = shape(problem, ident);
    let handle = T::sample::<R>(client, &tensor_shape, sample_seed);
    let data = client.read_one(handle.handle.binding());
    let data = T::from_bytes(&data);
    let original_data = data.to_owned();
    let data_bytes = T::as_bytes(&original_data);
    let shape = tensor_shape.as_slice();
    let elem_size = std::mem::size_of::<T>();
    let (handle, strides) = client.create_tensor(data_bytes, shape, elem_size);

    TensorRawParts {
        handle,
        shape: tensor_shape.to_vec(),
        strides,
        original_data: Some(original_data),
    }
}

fn tensor_raw_parts_output<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &AttentionProblem,
) -> TensorRawParts<P::EG> {
    let zero = P::EG::from_int(0);
    let data = vec![zero; tensor_size(problem, Ident::Out)];
    let tensor_shape = shape(problem, Ident::Out);
    let data_bytes = P::EG::as_bytes(&data);
    let shape = tensor_shape.as_slice();
    let elem_size = std::mem::size_of::<P::EG>();
    let (handle, strides) = client.create_tensor(data_bytes, shape, elem_size);

    TensorRawParts {
        handle,
        shape: tensor_shape.to_vec(),
        strides,
        original_data: None,
    }
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
