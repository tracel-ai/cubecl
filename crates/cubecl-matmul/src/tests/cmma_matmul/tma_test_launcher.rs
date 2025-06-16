use std::marker::PhantomData;

use cubecl_core::CubeElement;
use cubecl_core::prelude::*;

use crate::components::AvailableLineSizes;
use crate::components::Ident;
use crate::components::MatmulProblem;
use crate::components::MatrixLayout;
use crate::components::batch::BatchConfig;
use crate::components::batch::BatchMatmulFamily;
use crate::components::global::args::TensorMapArgs;
use crate::components::global::args::{ConcreteInputsFactory, TensorMapInputs};
use crate::kernels::matmul::Algorithm;
use crate::kernels::matmul::MatmulSelection;
use crate::tests::test_utils::Sample;
use crate::tests::test_utils::TestPrecision;

use super::matmul_test_launcher::{TensorRawParts, tensor_size, transpose};

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_tma_matmul_algorithm<A, P, R>(
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
    let lhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<P, R>(&client, &problem, Ident::Out);

    let elem_size = size_of::<P::EG>();
    let lhs_handle = TensorHandleRef {
        handle: &lhs.handle,
        strides: &lhs.strides,
        shape: &lhs.shape,
        elem_size,
        runtime: PhantomData,
    };
    let rhs_handle = TensorHandleRef {
        handle: &rhs.handle,
        strides: &rhs.strides,
        shape: &rhs.shape,
        elem_size,
        runtime: PhantomData,
    };

    let available_line_sizes = AvailableLineSizes::from_elem_types::<R>(
        &P::EG::as_elem_native_unchecked(),
        &P::EG::as_elem_native_unchecked(),
    );

    let config = match A::setup::<(P::EG, P::ES, P::EA, P::EG), R>(
        &client,
        &problem,
        &selection,
        available_line_sizes,
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

    let line_sizes = config.line_sizes();

    let inputs = TensorMapInputs::create(
        &lhs_handle,
        &None,
        &rhs_handle,
        &None,
        &selection,
        &problem,
        &line_sizes,
    );
    let output = unsafe {
        TensorArg::<R>::from_raw_parts::<P::EG>(
            &out.handle,
            &out.strides,
            &out.shape,
            line_sizes.out,
        )
    };

    unsafe {
        A::BatchMatmul::launch_unchecked::<((P::EG, P::ES, P::EA, P::EG), TensorMapArgs), R>(
            &client,
            config.cube_dim(),
            config.cube_count(&problem),
            inputs,
            output,
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
    problem: &MatmulProblem,
    ident: Ident,
) -> TensorRawParts<P::EG> {
    match ident {
        Ident::Lhs => {
            let mut shape = problem.shape(ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one(handle.handle.binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = shape.len();

            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    shape.swap(rank - 1, rank - 2);
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };

            let (handle, mut strides) =
                client.create_tensor(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

            if matches!(problem.lhs_layout, MatrixLayout::ColMajor) {
                shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: Some(original_data),
                quant_params: None,
            }
        }
        Ident::Rhs => {
            let mut shape = problem.shape(ident);

            let handle = P::EG::sample::<R>(client, &shape, 5678);

            let data = client.read_one(handle.handle.binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = shape.len();

            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    shape.swap(rank - 1, rank - 2);
                    transpose::<P::EG>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            let (handle, mut strides) =
                client.create_tensor(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

            if matches!(problem.rhs_layout, MatrixLayout::ColMajor) {
                shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: Some(original_data),
                quant_params: None,
            }
        }
        Ident::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, Ident::Out)];

            let shape = problem.shape(Ident::Out);
            let (handle, strides) =
                client.create_tensor(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());
            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: None,
                quant_params: None,
            }
        }
    }
}
