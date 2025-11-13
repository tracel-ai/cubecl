use cubecl_core::{CubeElement, server::Allocation};
use cubecl_core::{ir::StorageType, prelude::*};

use crate::components::batch::BatchConfig;
use crate::components::batch::BatchMatmulFamily;
use crate::components::global::args::TensorMapArgs;
use crate::components::global::args::{ConcreteInputsFactory, TensorMapInputs};
use crate::components::{AvailableLineSizes, MatrixLayout};
use crate::components::{MatmulElems, MatmulIdent};
use crate::components::{MatmulProblem, TilingScheme};
use crate::components::{MatmulSelection, stage::SwizzleMode};
use crate::kernels::layered::Algorithm;
use crate::tests::test_utils::Sample;
use crate::tests::test_utils::TestPrecision;
use crate::{
    MatmulInputHandleRef,
    components::global::args::{ConcreteOutputFactory, TensorOutput},
};

use super::matmul_test_launcher::{TensorRawParts, tensor_size, transpose};

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_tma_matmul_algorithm<A, P, R>(
    client: ComputeClient<R::Server>,
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

    let line_sizes = AvailableLineSizes::from_type_sizes::<R>(
        size_of::<P::EG>(),
        size_of::<P::EG>(),
        size_of::<P::EG>(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs(|ls| *ls == 1)
        .filter_rhs(|ls| *ls == 1)
        .pick_max()
        .unwrap();
    let dtypes = MatmulElems {
        lhs_global: P::EG::as_type_native_unchecked(),
        rhs_global: P::EG::as_type_native_unchecked(),
        acc_global: P::EA::as_type_native_unchecked(),
        lhs_stage: P::ES::as_type_native_unchecked(),
        rhs_stage: P::ES::as_type_native_unchecked(),
        acc_stage: P::EA::as_type_native_unchecked(),
        lhs_register: P::ES::as_type_native_unchecked(),
        rhs_register: P::ES::as_type_native_unchecked(),
        acc_register: P::EA::as_type_native_unchecked(),
    };
    let config = match A::setup::<R>(&client, &problem, &selection, &line_sizes, &dtypes) {
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

    let lhs = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<P, R>(&client, &problem, MatmulIdent::Out);

    let elem_size = size_of::<P::EG>();
    let lhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, elem_size)
        },
        P::EG::as_type_native_unchecked(),
    );
    let rhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, elem_size)
        },
        P::EG::as_type_native_unchecked(),
    );
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(&out.handle, &out.strides, &out.shape, elem_size)
    };

    let inputs = TensorMapInputs::create(
        &client,
        &lhs_handle,
        &rhs_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let output = TensorOutput::create(
        &client,
        &out_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    let dtypes = MatmulElems::new::<P::MP>();

    unsafe {
        A::BatchMatmul::launch_unchecked::<TensorMapArgs, R>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            inputs,
            output,
            cube_count_plan.as_args(),
            config,
            &dtypes,
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

// keep here for testing swizzling kernels, need to add macro for this eventually
#[allow(unused)]
fn select_swizzle(
    tiling: TilingScheme,
    ident: MatmulIdent,
    elem: StorageType,
    layout: MatrixLayout,
) -> SwizzleMode {
    let swizzle_dim = match layout {
        MatrixLayout::RowMajor => tiling.elements_in_stage_col(ident),
        MatrixLayout::ColMajor => tiling.elements_in_stage_row(ident),
    };
    let swizzle_dim_bytes = swizzle_dim as usize * elem.size();
    if !swizzle_dim_bytes.is_power_of_two() || swizzle_dim_bytes < 32 {
        return SwizzleMode::None;
    }
    match swizzle_dim_bytes {
        32 => SwizzleMode::B32,
        64 => SwizzleMode::B64,
        128 => SwizzleMode::B128,
        _ => SwizzleMode::None,
    }
}

fn tensor_raw_parts<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R::Server>,
    problem: &MatmulProblem,
    ident: MatmulIdent,
) -> TensorRawParts<P::EG> {
    match ident {
        MatmulIdent::Lhs => {
            let mut shape = problem.shape(ident);

            let handle = P::EG::sample::<R>(client, &shape, 1234);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
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

            let Allocation {
                handle,
                mut strides,
            } = client.create_tensor_from_slice(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

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
            }
        }
        MatmulIdent::Rhs => {
            let mut shape = problem.shape(ident);

            let handle = P::EG::sample::<R>(client, &shape, 5678);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
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

            let Allocation {
                handle,
                mut strides,
            } = client.create_tensor_from_slice(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

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
            }
        }
        MatmulIdent::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let shape = problem.shape(MatmulIdent::Out);
            let Allocation { handle, strides } =
                client.create_tensor_from_slice(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());
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
