use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::problem::MatmulProblem;
use crate::matmul::FixedShapeMatmul;
use crate::matmul::TensorMatmul;

use super::test_utils::assert_equals_approx;
use super::test_utils::matmul_cpu_reference;

pub fn test_fixed_matmul<MM, I, O, R>(layouts: (MatrixLayout, MatrixLayout), device: &R::Device)
where
    I: Numeric + CubeElement,
    O: Numeric + CubeElement,
    MM: FixedShapeMatmul<I, O>,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
    let problem = MatmulProblem {
        m: MM::M,
        n: MM::N,
        k: MM::K,
        lhs_layout: layouts.0,
        rhs_layout: layouts.1,
    };

    let lhs_size = (MM::M * MM::K) as usize;
    let rhs_size = (MM::K * MM::N) as usize;
    let out_size = (MM::M * MM::N) as usize;

    let lhs_original_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 1000.).collect();
    let rhs_original_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 1000.).collect();

    let lhs_data = match layouts.0 {
        MatrixLayout::RowMajor => lhs_original_data.clone(),
        MatrixLayout::ColMajor => {
            transpose::<f32>(&lhs_original_data, MM::M as usize, MM::K as usize)
        }
    };
    let rhs_data = match layouts.1 {
        MatrixLayout::RowMajor => rhs_original_data.clone(),
        MatrixLayout::ColMajor => {
            transpose::<f32>(&rhs_original_data, MM::K as usize, MM::N as usize)
        }
    };

    let lhs = client.create(I::as_bytes(&I::from_values(&lhs_data)));
    let rhs = client.create(I::as_bytes(&I::from_values(&rhs_data)));
    let out = client.empty(out_size * O::as_elem().size());

    let requirements = match MM::can_process(problem) {
        false => {
            panic!("Tried to test on a problem this algorithm can't solve")
        }
        true => MM::requirements(problem),
    };
    let cube_dim = CubeDim::new(32, requirements.num_planes, 1);
    let cube_count: CubeCount<<R as Runtime>::Server> =
        CubeCount::Static(requirements.num_cubes, 1, 1);

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            ArrayArg::<R>::from_raw_parts(&lhs, lhs_size, 1),
            ArrayArg::<R>::from_raw_parts(&rhs, rhs_size, 1),
            ArrayArg::<R>::from_raw_parts(&out, out_size, 1),
            layouts,
        );
    }

    let expected = matmul_cpu_reference(&lhs_original_data, &rhs_original_data, problem);
    if let Err(e) = assert_equals_approx::<O, R>(&client, out, &expected, 10e-1) {
        panic!("{}", e);
    }
}

pub fn test_tensor_matmul<MM, Elem, R>(problem: MatmulProblem, device: &R::Device)
where
    Elem: Numeric + CubeElement,
    MM: TensorMatmul<Elem>,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
    let lhs_size = problem.m * problem.k;
    let rhs_size = problem.k * problem.n;
    let out_size = problem.m * problem.n;

    let lhs_original_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 1000.).collect();
    let rhs_original_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 1000.).collect();

    let (lhs_data, lhs_strides) = match problem.lhs_layout {
        MatrixLayout::RowMajor => (lhs_original_data.clone(), [problem.k as usize, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&lhs_original_data, problem.m as usize, problem.k as usize),
            [1, problem.m as usize],
        ),
    };
    let (rhs_data, rhs_strides) = match problem.rhs_layout {
        MatrixLayout::RowMajor => (rhs_original_data.clone(), [problem.n as usize, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&rhs_original_data, problem.k as usize, problem.n as usize),
            [1, problem.k as usize],
        ),
    };

    let lhs = client.create(Elem::as_bytes(&Elem::from_values(&lhs_data)));
    let rhs = client.create(Elem::as_bytes(&Elem::from_values(&rhs_data)));
    let out = client.empty(out_size as usize * Elem::as_elem().size());

    let requirements = match MM::can_process(problem) {
        false => {
            panic!("Tried to test on a problem this algorithm can't solve")
        }
        true => MM::requirements(problem),
    };
    let cube_dim = CubeDim::new(32, requirements.num_planes, 1);
    let cube_count: CubeCount<<R as Runtime>::Server> =
        CubeCount::Static(requirements.num_cubes, 1, 1);

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(
                &lhs,
                &lhs_strides,
                &[problem.m as usize, problem.k as usize],
                1,
            ),
            TensorArg::<R>::from_raw_parts(
                &rhs,
                &rhs_strides,
                &[problem.k as usize, problem.n as usize],
                1,
            ),
            TensorArg::<R>::from_raw_parts(
                &out,
                &[problem.n as usize, 1],
                &[problem.m as usize, problem.n as usize],
                1,
            ),
            (problem.lhs_layout, problem.rhs_layout),
        );
    }

    let expected = matmul_cpu_reference(&lhs_original_data, &rhs_original_data, problem);
    if let Err(e) = assert_equals_approx::<Elem, R>(&client, out, &expected, 10e-1) {
        panic!("{}", e);
    }
}

fn transpose<E: Copy>(array: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut result = vec![array[0]; array.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = array[i * cols + j];
        }
    }
    result
}
