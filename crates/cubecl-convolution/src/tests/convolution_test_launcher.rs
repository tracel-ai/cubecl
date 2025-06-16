use cubecl_core::CubeElement;
use cubecl_core::prelude::*;
use cubecl_matmul::components::AvailableLineSizes;
use cubecl_matmul::components::global::GlobalConfig;
use cubecl_matmul::kernels::matmul::MatmulSelection;

use crate::ConvGemmConfig;
use crate::algorithm::Algorithm;
use crate::base::ConvolutionLaunch;
use crate::base::ConvolutionProblem;
use crate::{args::ConvInputsLaunch, base::ConvolutionConfigFactory};
use cubecl_matmul::components::InputIdent;
use cubecl_matmul::components::MatmulLineSizes;
use cubecl_matmul::components::global::args::{ConcreteOutputFactory, MatmulArgs};
use cubecl_matmul::tests::test_utils::Sample;
use cubecl_matmul::{components::Ident, tests::cmma_matmul::matmul_test_launcher::TensorRawParts};

use super::test_utils::TestPrecision;

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
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
    Args::Input<P::EG>: ConvInputsLaunch,
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
    let lhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Lhs);
    let rhs = tensor_raw_parts::<P, R>(&client, &problem, Ident::Rhs);
    let out = tensor_raw_parts::<P, R>(&client, &problem, Ident::Out);

    let available_line_sizes = AvailableLineSizes {
        lhs: vec![1],
        rhs: vec![1],
        out: R::line_size_elem(&P::EG::as_elem_native_unchecked()).collect(),
    };

    // let cube_dim = A::cube_dim(&selection);
    // let cube_count = A::cube_count(&selection, &problem);

    let config = match A::setup::<R, (P::EG, P::ES, f32, P::EG)>(
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

    let lhs_handle = A::into_tensor_handle::<R, P::EG>(&client, &lhs_handle, InputIdent::Lhs);
    let rhs_handle = A::into_tensor_handle::<R, P::EG>(&client, &rhs_handle, InputIdent::Rhs);

    let lhs_handle = lhs_handle.as_ref();
    let rhs_handle = rhs_handle.as_ref();

    let inputs = <Input<Args, P::EG> as ConvInputsLaunch>::create(
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
        A::GlobalConvolution::launch_unchecked::<((P::EG, P::ES, P::EA, P::EG), Args), R>(
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
    ident: Ident,
) -> TensorRawParts<P::EG> {
    match ident {
        Ident::Lhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one(handle.handle.clone().binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
                quant_params: None,
            }
        }
        Ident::Rhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one(handle.handle.clone().binding());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
                quant_params: None,
            }
        }
        Ident::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, Ident::Out)];

            let shape = shape(problem, Ident::Out);
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

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &ConvolutionProblem, ident: Ident) -> usize {
    match ident {
        Ident::Lhs => problem.m * problem.k,
        Ident::Rhs => problem.k * problem.n,
        Ident::Out => problem.m * problem.n,
    }
}

/// Returns the shape of the identified tensor, inferred by the problem definition
pub(crate) fn shape(problem: &ConvolutionProblem, ident: Ident) -> Vec<usize> {
    match ident {
        Ident::Lhs => vec![
            problem.batches,
            problem.shape[0],
            problem.shape[1],
            problem.channels,
        ],
        Ident::Rhs => vec![
            problem.n,
            problem.kernel_size[0] as usize,
            problem.kernel_size[1] as usize,
            problem.channels,
        ],
        Ident::Out => vec![
            problem.batches * problem.out_shape.iter().product::<usize>(),
            problem.n,
        ],
    }
}
