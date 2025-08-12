use cubecl::prelude::*;
use cubecl_core::Runtime;
use cubecl_core::{self as cubecl, server::AllocationDescriptor};
use cubecl_std::tensor::TensorHandle;

use alloc::{vec, vec::Vec};

use crate::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

pub fn test_quantization_tensor_symmetric<R: Runtime>(m: usize, n: usize) {
    let client = R::client(&Default::default());
    let shape = vec![m, n];

    let num_elems: usize = m * n;
    let data: Vec<_> = (0..num_elems).map(|v| v as f32).collect();
    let input_alloc = client.create_tensor(f32::as_bytes(&data), &shape, f32::elem_size() as usize);

    let q_max = i8::MAX as f32;
    let q_min = -q_max as f32;
    let range = num_elems as f32 - 1.0;
    let scale_f32 = (2.0 * range) / (q_max - q_min);
    let data_scale = vec![scale_f32];

    let scale_alloc =
        client.create_tensor(f32::as_bytes(&data_scale), &[1], f32::elem_size() as usize);

    let input = TensorHandle::<R, f32>::new(input_alloc.handle, shape, input_alloc.strides);
    let scale = TensorHandle::<R, f32>::new(scale_alloc.handle, vec![1], scale_alloc.strides);

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_value(QuantValue::QInt8)
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
        // We reuse the same buffer from the original input to store the
        // dequantized values.
        &input.as_ref(),
        &output_scale.as_ref(),
        &scheme,
    );

    let computed = client.read_one(input.handle);
    let data_restored = f32::from_bytes(&computed);

    assert_eq!(data_restored.len(), data.len());
    for (actual, expected) in data_restored.into_iter().zip(data.into_iter()) {
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

            #[test]
            fn test_quantization_tensor_symmetric_f32_32x32() {
                $crate::tests::test_quantization_tensor_symmetric::<TestRuntime>(32, 32);
            }
        }
    };
}
