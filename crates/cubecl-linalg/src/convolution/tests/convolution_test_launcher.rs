use cubecl_core::CubeElement;
use cubecl_core::prelude::*;
use cubecl_core::tensor_line_size_parallel;

use crate::convolution::base::ConvolutionProblem;
use crate::matmul::tests::test_utils::Sample;
use crate::matmul::{components::Ident, tests::cmma_matmul::matmul_test_launcher::TensorRawParts};
use crate::{convolution::algorithm::Algorithm, matmul::components::MatmulSelection};
use crate::{convolution::base::ConvolutionLaunch, matmul::components::InputIdent};
use crate::{
    convolution::{args::ConvInputsLaunch, base::ConvolutionConfigFactory},
    matmul::components::global::args::{ConcreteOutputFactory, MatmulArgs},
};

use super::test_utils::TestPrecision;

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_convolution_algorithm<A, Args, P, R>(
    client: ComputeClient<R::Server, R::Channel>,
    mut problem: ConvolutionProblem,
    input: <A::GlobalConvolution as ConvolutionConfigFactory>::Input,
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

    // No point vectorizing when we never deal with individual values anyways
    problem.lhs_line_size = 1;
    problem.rhs_line_size = 1;
    problem.out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&P::EG::as_elem_native_unchecked()),
        &out.shape,
        &out.strides,
        out.strides.len() - 1,
    );

    let cube_dim = A::cube_dim(&selection);
    let cube_count = A::cube_count(&selection, &problem);

    let config = match A::make_config::<R, (P::EG, P::ES, f32, P::EG)>(
        &client,
        input,
        &problem,
        &cube_dim,
        &cube_count,
    ) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {}", err);
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return;
            }
        }
    };

    if let Err(err) = A::check_availability::<R, (P::EG, P::ES, f32, P::EG)>(&client, &config) {
        let msg = format!("Skipped - not supported: {:?}", err);
        if panic_on_launch_err {
            panic!("{msg}")
        } else {
            println!("{msg}");
            client.flush();
            return;
        }
    }

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
    );
    let output = <Output<Args, P::EG> as ConcreteOutputFactory>::create(
        &out_handle,
        &selection,
        &problem.as_matmul_problem(),
    );

    unsafe {
        A::GlobalConvolution::launch_unchecked::<((P::EG, P::ES, P::EA, P::EG), Args), R>(
            &client, cube_dim, cube_count, inputs, None, output, &problem, config,
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
            let original_data = P::EG::sample(tensor_size(problem, Ident::Lhs), 1234);
            let shape = shape(problem, ident);

            let (handle, strides) =
                client.create_tensor(P::EG::as_bytes(&original_data), &shape, size_of::<P::EG>());

            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: Some(original_data),
            }
        }
        Ident::Rhs => {
            let original_data = P::EG::sample(tensor_size(problem, Ident::Rhs), 5678);
            let shape = shape(problem, ident);

            let (handle, strides) =
                client.create_tensor(P::EG::as_bytes(&original_data), &shape, size_of::<P::EG>());

            TensorRawParts {
                handle,
                shape,
                strides,
                original_data: Some(original_data),
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
                shape,
                strides,
                original_data: None,
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
            problem.height,
            problem.width,
            problem.channels,
        ],
        Ident::Rhs => vec![
            problem.n,
            problem.kernel_size.0 as usize,
            problem.kernel_size.1 as usize,
            problem.channels,
        ],
        Ident::Out => vec![problem.batches * problem.out_h * problem.out_w, problem.n],
    }
}
