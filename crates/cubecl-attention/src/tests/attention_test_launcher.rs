use cubecl_core::prelude::*;
use cubecl_core::server::Allocation;
use cubecl_core::{CubeElement, server};
use cubecl_std::CubeOptionArgs;

use crate::components::args::{TensorArgs, TensorInputsLaunch};
use crate::components::batch::BatchAttentionConfig;
use crate::components::batch::BatchAttentionFamily;
use crate::components::{AttentionElems, AttentionIdent, AvailableLineSizes};
use crate::components::{AttentionProblem, AttentionSelection};
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
    client: ComputeClient<R>,
    problem: AttentionProblem,
    selection: AttentionSelection,
) where
    A: Algorithm,
    P: TestPrecision,
    R: Runtime,
{
    let env = std::env::var("ATTENTION_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => matches!(val.as_str(), "panic"),
        Err(_) => false,
    };

    let query = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, AttentionIdent::Query, 12);
    let key = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, AttentionIdent::Key, 34);
    let value = tensor_raw_parts_input::<P, R, P::EG>(&client, &problem, AttentionIdent::Value, 56);
    let mask = match problem.masked {
        true => Some(tensor_raw_parts_input::<P, R, P::EM>(
            &client,
            &problem,
            AttentionIdent::Mask,
            78,
        )),
        false => None,
    };
    let out = tensor_raw_parts_output::<P, R>(&client, &problem);

    let attention_elems = AttentionElems::new::<P::AP>();
    let line_sizes = {
        let ls = AvailableLineSizes::from_elem_types(
            &client,
            attention_elems.query_global.size(),
            attention_elems.mask.size(),
            attention_elems.out_global.size(),
        );
        let ls = A::filter_line_sizes(ls)
            .filter_with_tensor(AttentionIdent::Query, &query.strides, &query.shape)
            .filter_with_tensor(AttentionIdent::Key, &key.strides, &key.shape)
            .filter_with_tensor(AttentionIdent::Value, &value.strides, &value.shape)
            .filter_with_tensor(AttentionIdent::Out, &out.strides, &out.shape);

        if let Some(mask) = mask.as_ref() {
            ls.filter_with_tensor(AttentionIdent::Mask, &mask.strides, &mask.shape)
        } else {
            ls
        }
    }
    .pick_max()
    .unwrap();

    let config = match A::setup(&client, &problem, &selection, &line_sizes, &attention_elems) {
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

    let cube_count_plan = config
        .hypercube_config()
        .cube_count_plan(&problem, &selection);

    unsafe {
        A::BatchAttention::launch_unchecked::<TensorArgs, R>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                TensorArg::from_raw_parts::<P::EG>(
                    &query.handle,
                    &query.strides,
                    &query.shape,
                    line_sizes.query,
                ),
                TensorArg::from_raw_parts::<P::EG>(
                    &key.handle,
                    &key.strides,
                    &key.shape,
                    line_sizes.key,
                ),
                TensorArg::from_raw_parts::<P::EG>(
                    &value.handle,
                    &value.strides,
                    &value.shape,
                    line_sizes.value,
                ),
                match mask.as_ref() {
                    Some(m) => CubeOptionArgs::Some(TensorArg::from_raw_parts::<P::EM>(
                        &m.handle,
                        &m.strides,
                        &m.shape,
                        line_sizes.mask,
                    )),
                    None => CubeOptionArgs::None,
                },
            ),
            TensorArg::from_raw_parts::<P::EG>(
                &out.handle,
                &out.strides,
                &out.shape,
                line_sizes.out,
            ),
            cube_count_plan.as_args(),
            config,
            &attention_elems,
        );
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

fn tensor_raw_parts_input<P: TestPrecision, R: Runtime, T>(
    client: &ComputeClient<R>,
    problem: &AttentionProblem,
    ident: AttentionIdent,
    sample_seed: u64,
) -> TensorRawParts<T>
where
    T: Numeric + CubeElement + Sampleable,
{
    let tensor_shape = problem.shape(ident);
    let handle = T::sample(client, &tensor_shape, sample_seed);
    let data = client.read_one(handle.handle);
    let data = T::from_bytes(&data);

    let original_data = data.to_owned();
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

fn tensor_raw_parts_output<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R>,
    problem: &AttentionProblem,
) -> TensorRawParts<P::EG> {
    let zero = P::EG::from_int(0);
    let data = vec![zero; tensor_size(problem, AttentionIdent::Out)];
    let tensor_shape = problem.shape(AttentionIdent::Out);
    let data_bytes = P::EG::as_bytes(&data);
    let shape = tensor_shape.as_slice();
    let elem_size = P::EG::type_size() as usize;
    let Allocation { handle, strides } =
        client.create_tensor_from_slice(data_bytes, shape, elem_size);

    TensorRawParts {
        handle,
        shape: tensor_shape.to_vec(),
        strides,
        original_data: None,
    }
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &AttentionProblem, ident: AttentionIdent) -> usize {
    problem.shape(ident).iter().product()
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
