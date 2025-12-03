use cubecl_core::prelude::*;
use cubecl_core::server::Allocation;
use cubecl_core::{CubeElement, server};
use cubecl_std::CubeOptionArgs;

use crate::components::AttentionIdent;
use crate::components::args::{TensorArgs, TensorInputsLaunch};
use crate::components::batch::BatchAttentionFamily;
use crate::components::{AttentionBlueprint, AttentionProblem};
use crate::kernels::Algorithm;
use crate::tests::test_utils::Sampleable;
use crate::tests::test_utils::TestPrecision;

use cubecl_core::{Runtime, client::ComputeClient};
use std::fmt::Debug;

// Returns if should return
fn should_abort<T, E: Debug>(result: &Result<T, E>) -> bool {
    let env = std::env::var("ATTENTION_TEST_MODE");
    let panic_on_error = env.as_deref() == Ok("panic");

    if let Err(err) = result {
        let msg = format!("Skipping the test with an execution error {err:?}");
        if panic_on_error {
            panic!("{msg}");
        } else {
            println!("{msg}");
        }

        true
    } else {
        false
    }
}

pub fn attention_test_launch<A: Algorithm, P: TestPrecision, R: Runtime>(
    client: ComputeClient<R>,
    problem: AttentionProblem,
    settings: &A::Settings,
) {
    let blueprint = A::blueprint(&client, &problem, settings);
    if should_abort(&blueprint) {
        return;
    }

    test_attention_algorithm::<A, P, R>(client, problem, blueprint.unwrap());
}

#[derive(Debug)]
pub struct TensorRawParts<N: Numeric + CubeElement> {
    pub handle: server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub original_data: Option<Vec<N>>,
}

enum TestInputType<T> {
    Random(u64),
    Data(Vec<T>),
    Zeros,
}

pub fn test_attention_algorithm<A, P, R>(
    client: ComputeClient<R>,
    problem: AttentionProblem,
    blueprint: AttentionBlueprint,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let query = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Query,
        TestInputType::Random(12),
    );
    let key = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Key,
        TestInputType::Random(34),
    );
    let value = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Value,
        TestInputType::Random(56),
    );
    let mask = if problem.masked {
        Some(tensor_raw_parts::<P::EM, R>(
            &client,
            &problem,
            AttentionIdent::Mask,
            TestInputType::Random(78),
        ))
    } else {
        None
    };
    let out =
        tensor_raw_parts::<P::EG, R>(&client, &problem, AttentionIdent::Out, TestInputType::Zeros);

    test_attention_algorithm_raw::<A, P, R>(
        client, problem, blueprint, query, key, value, mask, out,
    );
}

#[allow(unused)]
pub fn test_attention_algorithm_explicit<A, P, R>(
    client: ComputeClient<R>,
    problem: AttentionProblem,
    selection: AttentionBlueprint,
    query_data: Vec<P::EG>,
    key_data: Vec<P::EG>,
    value_data: Vec<P::EG>,
    mask_data: Option<Vec<P::EM>>,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let query = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Query,
        TestInputType::Data(query_data),
    );
    let key = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Key,
        TestInputType::Data(key_data),
    );
    let value = tensor_raw_parts::<P::EG, R>(
        &client,
        &problem,
        AttentionIdent::Value,
        TestInputType::Data(value_data),
    );
    let mask = mask_data.map(|m| {
        tensor_raw_parts::<P::EM, R>(
            &client,
            &problem,
            AttentionIdent::Mask,
            TestInputType::Data(m),
        )
    });
    let out =
        tensor_raw_parts::<P::EG, R>(&client, &problem, AttentionIdent::Out, TestInputType::Zeros);

    test_attention_algorithm_raw::<A, P, R>(
        client, problem, selection, query, key, value, mask, out,
    );
}

#[allow(clippy::too_many_arguments)]
fn test_attention_algorithm_raw<A, P, R>(
    client: ComputeClient<R>,
    problem: AttentionProblem,
    blueprint: AttentionBlueprint,
    query: TensorRawParts<P::EG>,
    key: TensorRawParts<P::EG>,
    value: TensorRawParts<P::EG>,
    mask: Option<TensorRawParts<P::EM>>,
    out: TensorRawParts<P::EG>,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let dtypes = A::dtypes(&client, &problem, &blueprint).unwrap();
    let cube_count_plan = blueprint.cube_count_plan(&problem);

    let result = unsafe {
        A::BatchAttention::launch_unchecked::<TensorArgs, R>(
            &client,
            blueprint.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                TensorArg::from_raw_parts::<P::EG>(
                    &query.handle,
                    &query.strides,
                    &query.shape,
                    blueprint.line_sizes.query,
                ),
                TensorArg::from_raw_parts::<P::EG>(
                    &key.handle,
                    &key.strides,
                    &key.shape,
                    blueprint.line_sizes.key,
                ),
                TensorArg::from_raw_parts::<P::EG>(
                    &value.handle,
                    &value.strides,
                    &value.shape,
                    blueprint.line_sizes.value,
                ),
                match mask.as_ref() {
                    Some(m) => CubeOptionArgs::Some(TensorArg::from_raw_parts::<P::EM>(
                        &m.handle,
                        &m.strides,
                        &m.shape,
                        blueprint.line_sizes.mask,
                    )),
                    None => CubeOptionArgs::None,
                },
            ),
            TensorArg::from_raw_parts::<P::EG>(
                &out.handle,
                &out.strides,
                &out.shape,
                blueprint.line_sizes.out,
            ),
            cube_count_plan.as_args(),
            &dtypes,
            blueprint,
        )
    };

    if should_abort(&result) {
        return;
    }

    P::assert_result(
        &query.original_data.unwrap(),
        &key.original_data.unwrap(),
        &value.original_data.unwrap(),
        mask.as_ref()
            .map(|m| m.original_data.as_ref().unwrap().as_slice()),
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );
}

fn tensor_raw_parts<T, R: Runtime>(
    client: &ComputeClient<R>,
    problem: &AttentionProblem,
    ident: AttentionIdent,
    input: TestInputType<T>,
) -> TensorRawParts<T>
where
    T: Numeric + CubeElement + Sampleable,
{
    let tensor_shape = problem.shape(ident);
    let original_data = match input {
        TestInputType::Random(seed) => {
            let handle = T::sample(client, &tensor_shape, seed);
            let data = client.read_one(handle.handle);
            let data = T::from_bytes(&data);
            data.to_owned()
        }
        TestInputType::Data(data) => {
            assert_eq!(
                data.len(),
                tensor_shape.iter().product::<usize>(),
                "Provided data length does not match tensor shape"
            );
            data
        }
        TestInputType::Zeros => vec![T::from_int(0); tensor_shape.len()],
    };

    let data_bytes = T::as_bytes(&original_data);
    let shape = tensor_shape.as_slice();
    let elem_size = T::type_size() as usize;

    let Allocation { handle, strides } =
        client.create_tensor_from_slice(data_bytes, shape, elem_size);

    TensorRawParts {
        handle,
        shape: tensor_shape.to_vec(),
        strides,
        original_data: Some(original_data),
    }
}

pub(crate) fn strides(problem: &AttentionProblem, ident: AttentionIdent) -> Vec<usize> {
    let shape = problem.shape(ident);

    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }

    strides
}
