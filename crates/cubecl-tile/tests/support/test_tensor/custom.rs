use cubecl::{
    client::Client,
    ir::{ElemType, FloatKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::support::BaseInputSpec;
use crate::support::test_tensor::strides::physical_extent;

fn new_custom_data(
    client: &Client,
    shape: Shape,
    strides: Strides,
    dtype: ElemType,
    contiguous_data: Vec<f32>,
) -> TensorHandle {
    let num_logical = shape.iter().product::<usize>();
    assert_eq!(
        contiguous_data.len(),
        num_logical,
        "DataKind::Custom expects data.len() == product(shape)",
    );

    let physical_size = physical_extent(&shape, &strides);

    let mut physical = vec![0.0f32; physical_size];
    scatter_logical_to_physical(&shape, &strides, &contiguous_data, &mut physical);

    let bytes = cast_f32_to_dtype(&physical, dtype);
    let handle = client.create_from_slice(&bytes);

    TensorHandle::new(handle, shape, strides, dtype)
}

/// Scatter logical-indexed `src` values to their physical offsets in `dst`.
///
/// When two logical indices map to the same physical offset (a broadcast
/// stride), the last write wins. Callers that care must supply values that
/// agree across the duplicate indices.
fn scatter_logical_to_physical(shape: &Shape, strides: &Strides, src: &[f32], dst: &mut [f32]) {
    let rank = shape.len();
    let mut coord = vec![0usize; rank];

    for (linear, &value) in src.iter().enumerate() {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % shape[d];
            rem /= shape[d];
        }
        let offset: usize = coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        dst[offset] = value;
    }
}

pub(crate) fn cast_f32_to_dtype(data: &[f32], dtype: ElemType) -> Vec<u8> {
    match dtype {
        ElemType::Float(FloatKind::F32) => f32::as_bytes(data).to_vec(),
        ElemType::Float(FloatKind::F64) => {
            let casted: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            f64::as_bytes(&casted).to_vec()
        }
        ElemType::Float(FloatKind::F16) => {
            let casted: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            half::f16::as_bytes(&casted).to_vec()
        }
        ElemType::Float(FloatKind::BF16) => {
            let casted: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
            half::bf16::as_bytes(&casted).to_vec()
        }
        ElemType::UInt(cubecl::ir::UIntKind::U32) => {
            let casted: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            u32::as_bytes(&casted).to_vec()
        }
        ElemType::UInt(cubecl::ir::UIntKind::U8) => data.iter().map(|&x| x as u8).collect(),
        ElemType::Int(cubecl::ir::IntKind::I32) => {
            let casted: Vec<i32> = data.iter().map(|&x| x as i32).collect();
            i32::as_bytes(&casted).to_vec()
        }
        ElemType::Int(cubecl::ir::IntKind::I8) => {
            let casted: Vec<i8> = data.iter().map(|&x| x as i8).collect();
            i8::as_bytes(&casted).to_vec()
        }
        other => panic!("cast_f32_to_dtype: unsupported storage type {other:?}"),
    }
}

pub(crate) fn build_custom(base_spec: BaseInputSpec, contiguous_data: Vec<f32>) -> TensorHandle {
    new_custom_data(
        &base_spec.client,
        base_spec.shape.clone(),
        base_spec.strides(),
        base_spec.dtype,
        contiguous_data,
    )
}
