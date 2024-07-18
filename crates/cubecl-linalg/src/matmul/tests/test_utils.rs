use bytemuck::cast_slice;
use cubecl_core::{
    frontend::{F16, F32},
    ir::{Elem, FloatKind},
    server::Handle,
    CubeElement, Feature, Runtime,
};
use std::ops::Range;

use crate::{
    matmul::tiling2d::config::{CubeTiling2dConfig, Tiling2dConfig},
    tensor::TensorHandle,
};

pub(crate) fn range_tensor_f16<R: Runtime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> TensorHandle<R, F16> {
    let n_elements = x * y;
    let client = R::client(device);

    let mut data = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        data.push(half::f16::from_f32(i as f32));
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn range_tensor<R: Runtime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> TensorHandle<R, F32> {
    let n_elements = x * y;
    let client = R::client(device);

    let mut data: Vec<f32> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        data.push(i as f32);
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn range_tensor_with_factor<R: Runtime>(
    batch: usize,
    x: usize,
    y: usize,
    factor: f32,
    device: &R::Device,
) -> TensorHandle<R, F32> {
    let n_elements = batch * x * y;
    let client = R::client(device);

    let mut data: Vec<f32> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        data.push(i as f32 / factor);
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![batch, x, y], handle)
}

pub(crate) fn range_tensor_transposed<R: Runtime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> TensorHandle<R, F32> {
    let n_elements = x * y;
    let client = R::client(device);

    let mut data: Vec<f32> = Vec::with_capacity(n_elements);
    for i in 0..y {
        for j in 0..x {
            let number = j * y + i;
            data.push(number as f32);
        }
    }

    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn zeros_tensor<R: Runtime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> TensorHandle<R, F32> {
    let n_elements = x * y;
    let client = R::client(device);

    let data: Vec<f32> = vec![0.; n_elements];
    let handle = client.create(cast_slice(&data));

    TensorHandle::new_contiguous(vec![x, y], handle)
}

pub(crate) fn create_empty<R: Runtime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> Handle<<R as Runtime>::Server> {
    let client = R::client(device);
    client.empty(x * y * core::mem::size_of::<f32>())
}

pub(crate) fn assert_equals<R: Runtime>(
    output: Handle<<R as Runtime>::Server>,
    expected: &[f32],
    device: &R::Device,
) {
    let client = R::client(device);

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, expected);
}

pub(crate) fn assert_equals_approx<R: Runtime>(
    output: Handle<<R as Runtime>::Server>,
    expected: &[f32],
    epsilon: f32,
    device: &R::Device,
) {
    let client = R::client(device);

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < epsilon,
            "Values differ more than epsilon: actual={}, expected={}, difference={}, epsilon={}",
            a,
            e,
            (a - e).abs(),
            epsilon
        );
    }
}

pub(crate) fn assert_equals_range<R: Runtime>(
    output: Handle<<R as Runtime>::Server>,
    expected: &[f32],
    range: Range<usize>,
    device: &R::Device,
) {
    let client = R::client(device);

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(&actual[range], expected);
}

pub(crate) fn make_tiling2d_config(m: usize, k: usize, n: usize) -> CubeTiling2dConfig {
    let tiling2d_config = Tiling2dConfig {
        block_size_m: 8,
        block_size_k: 8,
        block_size_n: 8,
        ..Default::default()
    };
    CubeTiling2dConfig::new(&tiling2d_config, m, k, n, false, false)
}

pub(crate) fn cmma_available<R: Runtime>(device: &R::Device) -> bool {
    R::client(device).features().enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    })
}
