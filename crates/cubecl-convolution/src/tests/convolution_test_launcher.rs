use cubecl_core::prelude::*;
use cubecl_core::{CubeElement, server::Allocation};
use cubecl_matmul::components::AvailableLineSizes;
use cubecl_matmul::components::MatmulIdent;
use cubecl_matmul::components::MatmulSelection;
use cubecl_matmul::components::global::GlobalConfig;

use crate::ConvGemmConfig;
use crate::algorithm::Algorithm;
use crate::args::ConvInputsLaunch;
use crate::base::ConvolutionLaunch;
use crate::base::ConvolutionProblem;
use cubecl_matmul::components::global::args::{ConcreteOutputFactory, MatmulArgs};
use cubecl_matmul::tests::layered::matmul_test_launcher::TensorRawParts;
use cubecl_matmul::tests::test_utils::Sample;

use super::test_utils::TestPrecision;

type Input<Args, Lhs, Rhs> = <Args as MatmulArgs>::Input<Lhs, Rhs>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_convolution_algorithm<A, Args, P, R>(
    client: ComputeClient<R::Server, R::Channel>,
    problem: ConvolutionProblem,
    selection: MatmulSelection,
) where
    A: Algorithm,
    Args: MatmulArgs,
    P: TestPrecision,
    R: Runtime,
    Args::Input<P::EG, P::EG>: ConvInputsLaunch,
    Args::Output<P::EG>: ConcreteOutputFactory,
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

    let line_sizes = AvailableLineSizes {
        lhs: vec![1],
        rhs: vec![1],
        out: R::line_size_elem(&P::EG::as_elem_native_unchecked()).collect(),
    }
    .filter_lhs_with_tensor(&lhs.strides, &lhs.shape, problem.lhs_layout)
    .filter_rhs_with_tensor(&rhs.strides, &rhs.shape, problem.rhs_layout)
    .filter_out_with_tensor(&out.strides, &out.shape)
    .pick_max()
    .unwrap();

    let config = match A::setup::<R, (P::EG, P::EG, P::ES, P::ES, f32, P::EG)>(
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

    let elem_size = size_of::<P::EG>();
    let lhs_handle = unsafe {
        TensorHandleRef::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, elem_size)
    };
    let rhs_handle = unsafe {
        TensorHandleRef::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, elem_size)
    };
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(&out.handle, &out.strides, &out.shape, elem_size)
    };

    let lhs_handle = A::into_tensor_handle::<R, P::EG>(&client, &lhs_handle, MatmulIdent::Lhs);
    let rhs_handle = A::into_tensor_handle::<R, P::EG>(&client, &rhs_handle, MatmulIdent::Rhs);

    let lhs_handle = lhs_handle.as_ref();
    let rhs_handle = rhs_handle.as_ref();

    let inputs = <Input<Args, P::EG, P::EG> as ConvInputsLaunch>::create(
        &lhs_handle,
        &rhs_handle,
        &selection,
        &problem,
        &config.line_sizes(),
    );
    let output = <Output<Args, P::EG> as ConcreteOutputFactory>::create(
        &out_handle,
        &selection,
        &problem.as_matmul_problem(),
        &config.line_sizes(),
    );

    unsafe {
        A::GlobalConvolution::launch_unchecked::<
            ((P::EG, P::EG, P::ES, P::ES, P::EA, P::EG), Args),
            R,
        >(
            &client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            inputs,
            None,
            output,
            &problem,
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
    problem: &ConvolutionProblem,
    ident: MatmulIdent,
) -> TensorRawParts<P::EG> {
    match ident {
        MatmulIdent::Lhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one(handle.handle.clone());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Rhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one(handle.handle.clone());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let shape = shape(problem, MatmulIdent::Out);
            let Allocation { handle, strides } =
                client.create_tensor(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: None,
            }
        }
    }
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &ConvolutionProblem, ident: MatmulIdent) -> usize {
    match ident {
        MatmulIdent::Lhs => problem.m * problem.k,
        MatmulIdent::Rhs => problem.k * problem.n,
        MatmulIdent::Out => problem.m * problem.n,
    }
}

/// Returns the shape of the identified tensor, inferred by the problem definition
pub(crate) fn shape(problem: &ConvolutionProblem, ident: MatmulIdent) -> Vec<usize> {
    match ident {
        MatmulIdent::Lhs => vec![
            problem.batches,
            problem.shape[0],
            problem.shape[1],
            problem.channels,
        ],
        MatmulIdent::Rhs => vec![
            problem.n,
            problem.kernel_size[0] as usize,
            problem.kernel_size[1] as usize,
            problem.channels,
        ],
        MatmulIdent::Out => vec![
            problem.batches * problem.out_shape.iter().product::<usize>(),
            problem.n,
        ],
    }
}
