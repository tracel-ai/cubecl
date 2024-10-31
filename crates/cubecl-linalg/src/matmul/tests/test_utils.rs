use std::fmt::Display;

use bytemuck::cast_slice;
use cubecl_core::{client::ComputeClient, prelude::Float, server::Handle, CubeElement, Runtime};

use crate::{
    matmul::tiling2d::config::{CubeTiling2dConfig, Tiling2dConfig},
    tensor::TensorHandle,
};

pub(crate) fn range_tensor<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    x: usize,
    y: usize,
) -> TensorHandle<R, F> {
    let n_elements = x * y;

    let mut data: Vec<F> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        data.push(F::new(i as f32));
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn range_tensor_transposed<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    x: usize,
    y: usize,
) -> TensorHandle<R, F> {
    let n_elements = x * y;

    let mut data: Vec<F> = Vec::with_capacity(n_elements);
    for i in 0..y {
        for j in 0..x {
            let number = j * y + i;
            data.push(F::new(number as f32));
        }
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn zeros_tensor<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    x: usize,
    y: usize,
) -> TensorHandle<R, F> {
    let n_elements = x * y;

    let data: Vec<F> = vec![F::new(0.); n_elements];
    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn create_empty<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    x: usize,
    y: usize,
) -> Handle {
    client.empty(x * y * core::mem::size_of::<F>())
}

pub(crate) fn assert_equals<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
) {
    let actual = client.read(output.binding());
    let actual = F::from_bytes(&actual);

    pretty_assertions::assert_eq!(&actual[0..expected.len()], expected);
}

pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read(output.binding());
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

        if f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error {
            return Err(format!(
            "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
            i,
            *a,
            *e,
            f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
            epsilon
            ));
        }
    }

    Ok(())
}

pub fn make_tiling2d_config(m: usize, k: usize, n: usize) -> CubeTiling2dConfig {
    let tiling2d_config = Tiling2dConfig {
        block_size_m: 8,
        block_size_k: 8,
        block_size_n: 8,
        ..Default::default()
    };
    CubeTiling2dConfig::new(&tiling2d_config, m, k, n, false, false)
}

pub(crate) fn random_tensor<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: Vec<usize>,
) -> TensorHandle<R, F> {
    let data: Vec<F> = generate_random_data(shape.iter().product());
    let handle = client.create(cast_slice(&data));
    TensorHandle::new_contiguous(shape, handle)
}

pub(crate) fn generate_random_data<F: Float + CubeElement>(num_elements: usize) -> Vec<F> {
    fn lcg(seed: &mut u64) -> f32 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: f64 = 2u64.pow(32) as f64;

        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
        (*seed as f64 / M * 2.0 - 1.0) as f32
    }

    let mut seed = 12345;

    (0..num_elements).map(|_| F::new(lcg(&mut seed))).collect()
}
