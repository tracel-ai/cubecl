use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix_layout::MatrixLayout;
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
    let lhs_size = (MM::M * MM::K) as usize;
    let rhs_size = (MM::K * MM::N) as usize;
    let out_size = (MM::M * MM::N) as usize;

    let lhs_original_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 100.).collect();
    let rhs_original_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 100.).collect();

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

    let cube_dim = MM::cube_dim_resources();
    let cube_count: CubeCount<<R as Runtime>::Server> = MM::cube_count_resources();

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

    let expected = matmul_cpu_reference(
        &lhs_original_data,
        &rhs_original_data,
        MM::M as usize,
        MM::N as usize,
        MM::K as usize,
    );
    if let Err(e) = assert_equals_approx::<O, R>(&client, out, &expected, 10e-1) {
        panic!("{}", e);
    }
}

pub fn test_tensor_matmul<MM, Elem, R>(
    m: usize,
    n: usize,
    k: usize,
    layouts: (MatrixLayout, MatrixLayout),
    device: &R::Device,
) where
    Elem: Numeric + CubeElement,
    MM: TensorMatmul<Elem>,
    R: Runtime,
{
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
    let lhs_size = m * k;
    let rhs_size = k * n;
    let out_size = m * n;

    let lhs_original_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 100.).collect();
    let rhs_original_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 100.).collect();

    let (lhs_data, lhs_strides) = match layouts.0 {
        MatrixLayout::RowMajor => (lhs_original_data.clone(), [k, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&lhs_original_data, m as usize, k as usize),
            [1, m],
        ),
    };
    let (rhs_data, rhs_strides) = match layouts.1 {
        MatrixLayout::RowMajor => (rhs_original_data.clone(), [n, 1]),
        MatrixLayout::ColMajor => (
            transpose::<f32>(&rhs_original_data, k as usize, n as usize),
            [1, k],
        ),
    };

    let lhs = client.create(Elem::as_bytes(&Elem::from_values(&lhs_data)));
    let rhs = client.create(Elem::as_bytes(&Elem::from_values(&rhs_data)));
    let out = client.empty(out_size * Elem::as_elem().size());

    let cube_dim = MM::cube_dim_resources();
    let cube_count: CubeCount<<R as Runtime>::Server> = MM::cube_count_resources();

    unsafe {
        MM::launch_unchecked(
            &client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(&lhs, &lhs_strides, &[m, k], 1),
            TensorArg::<R>::from_raw_parts(&rhs, &rhs_strides, &[k, n], 1),
            TensorArg::<R>::from_raw_parts(&out, &[n, 1], &[m, n], 1),
            layouts,
        );
    }

    let expected = matmul_cpu_reference(&lhs_original_data, &rhs_original_data, m, n, k);
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
