use cubecl::prelude::*;
use cubecl_core::Runtime;
use cubecl_core::server::CopyDescriptor;
use cubecl_core::{self as cubecl, server::AllocationDescriptor};
use cubecl_std::tensor::TensorHandle;

use alloc::{vec, vec::Vec};

use crate::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

pub fn test_quantization_tensor<R: Runtime>(m: usize, n: usize, value: QuantValue) {
    let mode = QuantMode::Symmetric;
    let client = R::client(&Default::default());
    let shape = vec![m, n];

    let num_elems: usize = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<_> = (0..num_elems).map(|v| v as f32 - half).collect();
    let input_alloc = client.create_tensor(f32::as_bytes(&data), &shape, f32::elem_size() as usize);

    let (q_min, q_max) = value.range();
    let range = match value.is_symmetric() {
        true => num_elems as f32 - 1.0,
        false => num_elems as f32,
    };
    let scale_f32 = (2.0 * range) / (q_max - q_min);
    let data_scale = vec![scale_f32];

    let scale_alloc =
        client.create_tensor(f32::as_bytes(&data_scale), &[1], f32::elem_size() as usize);

    let input = TensorHandle::<R, f32>::new(input_alloc.handle, shape, input_alloc.strides);
    println!("{input:?}");
    let scale = TensorHandle::<R, f32>::new(scale_alloc.handle, vec![1], scale_alloc.strides);

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

    println!("{output:?}");
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
        // We reuse the same buffer from the original input to store the
        // dequantized values.
        &input.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    let computed = client.read_one_tensor(CopyDescriptor::new(
        input.handle.binding(),
        &input.shape,
        &input.strides,
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    assert_eq!(data_restored.len(), data.len());
    for (actual, expected) in data_restored.iter().zip(data.into_iter()) {
        let diff = f32::abs(actual - expected);
        assert!(diff <= scale_f32);
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quant {
    () => {
        mod quant {
            use super::*;
            use cubecl::prelude::*;
            use cubecl::{client::ComputeClient, prelude::TensorHandleRef};
            use cubecl_core as cubecl;
            use cubecl_quant::scheme::{QuantMode, QuantValue};

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q8f() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q8F);
            }

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q4f() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q4F);
            }

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q2f() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q2F);
            }

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q8s() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q8S);
            }

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q4s() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q4S);
            }

            #[test]
            fn test_quantization_tensor_f32_32x32_symmetric_q2s() {
                $crate::tests::test_quantization_tensor::<TestRuntime>(32, 32, QuantValue::Q2S);
            }
        }
    };
}
