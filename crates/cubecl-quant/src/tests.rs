use cubecl::prelude::*;
use cubecl_core::Runtime;
use cubecl_core::server::CopyDescriptor;
use cubecl_core::{self as cubecl, server::AllocationDescriptor};
use cubecl_std::tensor::TensorHandle;

use alloc::{vec, vec::Vec};

use crate::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

pub fn test_quantization_tensor_symmetric<R: Runtime>(m: usize, n: usize, value: QuantValue) {
    let mode = QuantMode::Symmetric;
    let client = R::client(&Default::default());
    let shape = vec![m, n];

    let num_elems: usize = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<_> = (0..num_elems).map(|v| v as f32 - half).collect();
    let input_alloc = client.create_tensor(f32::as_bytes(&data), &shape, f32::elem_size() as usize);

    let (q_min, q_max) = value.range();
    // input data range is not affected by quant range symmetry
    let scale_f32 = (2.0 * half) / (q_max - q_min);
    let data_scale = vec![scale_f32];

    let scale_alloc =
        client.create_tensor(f32::as_bytes(&data_scale), &[1], f32::elem_size() as usize);

    let input = TensorHandle::<R, f32>::new(input_alloc.handle, shape.clone(), input_alloc.strides);
    let scale = TensorHandle::<R, f32>::new(scale_alloc.handle, vec![1], scale_alloc.strides);
    let output_f = TensorHandle::<R, f32>::zeros(&client, shape);

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_mode(mode)
        .with_value(value)
        .with_store(QuantStore::U32)
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    // The shape is from the POV of packed u32s.
    let shape_out = vec![m, n / scheme.num_quants()];

    let [output_alloc, output_scale_alloc] = client
        .empty_tensors(vec![
            AllocationDescriptor {
                kind: cubecl_core::server::AllocationKind::Contiguous,
                shape: &shape_out,
                elem_size: u32::elem_size() as usize,
            },
            AllocationDescriptor {
                kind: cubecl_core::server::AllocationKind::Contiguous,
                shape: &[1],
                elem_size: f32::elem_size() as usize,
            },
        ])
        .try_into()
        .unwrap();
    let output = TensorHandle::<R, u32>::new(output_alloc.handle, shape_out, output_alloc.strides);
    let output_scale = TensorHandle::<R, f32>::new(
        output_scale_alloc.handle,
        vec![1],
        output_scale_alloc.strides,
    );

    crate::quantize::launch_ref::<R, f32>(
        &client,
        &input.as_ref(),
        &output.as_ref(),
        &scale.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    crate::dequantize::launch_ref::<R, f32>(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        &output.as_ref(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        &output_f.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    let computed = client.read_one_tensor(CopyDescriptor::new(
        output_f.handle.binding(),
        &output_f.shape,
        &output_f.strides,
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    // Max quantization error = step size / 2
    let rel_tol = 1e-4;
    let max_error = (scale_f32 / 2.0) * (1f32 + rel_tol);
    assert_eq!(data_restored.len(), data.len());
    for (actual, expected) in data_restored.iter().zip(data.into_iter()) {
        let diff = f32::abs(actual - expected);
        assert!(
            diff <= max_error,
            "Expected: {expected} | Actual: {actual} (diff {diff} > {max_error})"
        );
    }
}

pub fn test_quantization_block_symmetric<R: Runtime>(
    m: usize,
    n: usize,
    value: QuantValue,
    block_size: usize,
) {
    let mode = QuantMode::Symmetric;
    let client = R::client(&Default::default());
    let shape = vec![m, n];

    let num_elems: usize = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<_> = (0..num_elems)
        .map(|v| (v as f32 - half) / num_elems as f32)
        .collect();
    let input_alloc = client.create_tensor(f32::as_bytes(&data), &shape, f32::elem_size() as usize);

    let (q_min, q_max) = value.range();

    let scale_count = data.len() / block_size;
    let shape_scale = vec![m, n / block_size];

    let mut scales = Vec::with_capacity(scale_count);

    for block in 0..scale_count {
        let mut c_max = f32::MIN;
        let mut c_min = f32::MAX;

        let block_offset = block * block_size;

        for i in 0..block_size {
            let current = data[block_offset + i];
            c_max = f32::max(c_max, current);
            c_min = f32::min(c_min, current);
        }

        // The bias is assumed to be zero.
        let range = 2.0 * c_min.abs().max(c_max.abs());
        let scale = range / (q_max - q_min);
        scales.push(scale);
    }

    let scale_alloc = client.create_tensor(
        f32::as_bytes(&scales),
        &shape_scale,
        f32::elem_size() as usize,
    );

    let input = TensorHandle::<R, f32>::new(input_alloc.handle, shape.clone(), input_alloc.strides);
    let scale =
        TensorHandle::<R, f32>::new(scale_alloc.handle, shape_scale.clone(), scale_alloc.strides);
    let output_f = TensorHandle::<R, f32>::zeros(&client, shape);

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([block_size as u8]))
        .with_mode(mode)
        .with_value(value)
        .with_store(QuantStore::U32)
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    // The shape is from the POV of packed u32s.
    let shape_out = vec![m, n / scheme.num_quants()];

    let [output_alloc, output_scale_alloc] = client
        .empty_tensors(vec![
            AllocationDescriptor {
                kind: cubecl_core::server::AllocationKind::Contiguous,
                shape: &shape_out,
                elem_size: u32::elem_size() as usize,
            },
            AllocationDescriptor {
                kind: cubecl_core::server::AllocationKind::Contiguous,
                shape: &shape_scale,
                elem_size: f32::elem_size() as usize,
            },
        ])
        .try_into()
        .unwrap();
    let output = TensorHandle::<R, u32>::new(output_alloc.handle, shape_out, output_alloc.strides);
    let output_scale = TensorHandle::<R, f32>::new(
        output_scale_alloc.handle,
        shape_scale.clone(),
        output_scale_alloc.strides,
    );

    crate::quantize::launch_ref::<R, f32>(
        &client,
        &input.as_ref(),
        &output.as_ref(),
        &scale.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    crate::dequantize::launch_ref::<R, f32>(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        &output.as_ref(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        &output_f.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    let computed = client.read_one_tensor(CopyDescriptor::new(
        output_f.handle.binding(),
        &output_f.shape,
        &output_f.strides,
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    assert_eq!(data_restored.len(), data.len());
    let rel_tol = 1e-4;
    for (i, (actual, expected)) in data_restored.iter().zip(data.into_iter()).enumerate() {
        let block = i / block_size;
        let scale = scales[block];
        // Max quantization error = step size / 2
        let max_error = (scale / 2.0) * (1f32 + rel_tol);
        let diff = f32::abs(actual - expected);
        assert!(
            diff <= max_error,
            "Mismatch at {i}, Expected: {expected} | Actual: {actual} (diff {diff} > {max_error})"
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quant {
    ($value: expr, $shape_x: expr, $shape_y: expr) => {
        #[test]
        fn test_quantization_symmetric_tensor() {
            $crate::tests::test_quantization_tensor_symmetric::<TestRuntime>(
                $shape_x, $shape_y, $value,
            );
        }

        #[test]
        fn test_quantization_symmetric_block() {
            $crate::tests::test_quantization_block_symmetric::<TestRuntime>(
                $shape_x, $shape_y, $value, $shape_x, // Shape x as block_size
            );
        }
    };
    ($shape_x: expr, $shape_y: expr) => {
        mod q8f {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q8F, $shape_x, $shape_y);
        }
        mod q8s {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q8S, $shape_x, $shape_y);
        }
        mod q4f {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q4F, $shape_x, $shape_y);
        }
        mod q4s {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q4S, $shape_x, $shape_y);
        }
        mod q2f {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q2F, $shape_x, $shape_y);
        }
        mod q2s {
            use super::*;
            cubecl_quant::testgen_quant!(QuantValue::Q2S, $shape_x, $shape_y);
        }
    };
    () => {
        mod quant {
            use super::*;
            use cubecl::prelude::*;
            use cubecl::{client::ComputeClient, prelude::TensorHandleRef};
            use cubecl_core as cubecl;
            use cubecl_quant::scheme::{QuantMode, QuantValue};

            mod size32x32 {
                use super::*;
                cubecl_quant::testgen_quant!(32, 32);
            }
            mod size16x64 {
                use super::*;
                cubecl_quant::testgen_quant!(16, 64);
            }
        }
    };
}
