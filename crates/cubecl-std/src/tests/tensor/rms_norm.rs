use core::{any::TypeId, mem::size_of};

use cubecl_core::{
    CubeElement,
    prelude::{Float, Runtime},
};
use half::{bf16, f16};

use crate::tensor::{self, TensorHandle};

/// Reference RMS normalization used to validate the GPU implementation.
fn rms_norm_cpu<F: Copy + Into<f32>>(
    input: &[F],
    weight: &[F],
    bias: Option<&[F]>,
    shape: &[usize],
    epsilon: f32,
) -> Vec<f32> {
    let axis_size = *shape.last().expect("shape must not be empty");
    if axis_size == 0 {
        return vec![0.0; input.len()];
    }
    let rows = input.len() / axis_size;
    let mut output = vec![0.0; input.len()];

    for row in 0..rows {
        let start = row * axis_size;
        let mut sum_sq = 0.0f32;
        for idx in 0..axis_size {
            let value: f32 = input[start + idx].into();
            sum_sq += value * value;
        }
        let inv_rms = 1.0 / (sum_sq / axis_size as f32 + epsilon).sqrt();

        for idx in 0..axis_size {
            let mut value = input[start + idx].into() * inv_rms * weight[idx].into();
            if let Some(bias_slice) = bias {
                value += bias_slice[idx].into();
            }
            output[start + idx] = value;
        }
    }

    output
}

/// Execute RMS normalization on the provided runtime and compare the output with the CPU
/// reference. The last dimension of `shape` is normalized.
pub fn test_rms_norm<R, F>(device: &R::Device, shape: &[usize], epsilon: f32, with_bias: bool)
where
    R: Runtime,
    F: Float + CubeElement + Copy + Into<f32>,
{
    let client = R::client(device);
    let total_elems: usize = shape.iter().product();
    let axis_size = *shape.last().expect("shape must not be empty");

    let input: Vec<F> = (0..total_elems)
        .map(|idx| {
            let base = (idx as f32 * 0.137_f32).sin() + (idx % (axis_size.max(1))) as f32 * 0.021;
            F::new(base)
        })
        .collect();

    let weight: Vec<F> = (0..axis_size)
        .map(|idx| {
            let value = 0.75_f32 + (idx as f32 % 11.0) * 0.02;
            F::new(value)
        })
        .collect();

    let bias_values: Option<Vec<F>> = if with_bias {
        Some(
            (0..axis_size)
                .map(|idx| {
                    let value = -0.1_f32 + (idx as f32 % 7.0) * 0.015;
                    F::new(value)
                })
                .collect(),
        )
    } else {
        None
    };

    let expected = rms_norm_cpu(
        &input,
        &weight,
        bias_values.as_ref().map(|vec| &vec[..]),
        shape,
        epsilon,
    );

    let input_allocation = client.create_tensor(F::as_bytes(&input), shape, size_of::<F>());
    let input_handle = TensorHandle::<R, F>::new(
        input_allocation.handle,
        shape.to_vec(),
        input_allocation.strides,
    );

    let weight_allocation =
        client.create_tensor(F::as_bytes(&weight), &[axis_size], size_of::<F>());
    let weight_handle = TensorHandle::<R, F>::new(
        weight_allocation.handle,
        vec![axis_size],
        weight_allocation.strides,
    );

    let bias_handle = bias_values.as_ref().map(|bias_vec| {
        let allocation = client.create_tensor(F::as_bytes(bias_vec), &[axis_size], size_of::<F>());
        TensorHandle::<R, F>::new(allocation.handle, vec![axis_size], allocation.strides)
    });

    let output_handle = TensorHandle::<R, F>::empty(&client, shape.to_vec());

    tensor::rms_norm::launch(
        &client,
        &input_handle,
        &weight_handle,
        bias_handle.as_ref(),
        &output_handle,
        epsilon,
    );

    let actual_bytes = client.read_one_tensor(output_handle.as_copy_descriptor());
    let actual: Vec<f32> = F::from_bytes(&actual_bytes)
        .iter()
        .copied()
        .map(|value| value.into())
        .collect();

    let type_id = TypeId::of::<F>();
    let tolerance = if type_id == TypeId::of::<f16>() {
        6e-2
    } else if type_id == TypeId::of::<bf16>() {
        8e-2
    } else {
        1e-5
    };

    for (idx, (expected_value, actual_value)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (expected_value - actual_value).abs();
        assert!(
            diff <= tolerance,
            "RMSNorm mismatch at position {idx}: expected {expected_value}, got {actual_value}, diff {diff}"
        );
    }
}
