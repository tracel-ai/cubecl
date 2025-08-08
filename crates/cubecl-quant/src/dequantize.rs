#![allow(missing_docs)] // pub cube modules

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};

use crate::{
    qparams::QParams,
    scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
    utils::strided_layout,
};
use cubecl_std::tensor::{StridedLayout, index_offset_contiguous};
use half::{bf16, f16};

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: Float>(value: Line<F>, scale: FS) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * value
}

// TODO: use for fusion @nath

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_values<F: Float, FS: Float, QI: Int>(
    position: u32,
    values: &Tensor<QI>,
    scales: &Tensor<FS>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let value = values[position];
    dequantize_packed_value_at::<F, FS, QI>(position, value, scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value_at<F: Float, FS: Float, QI: Int>(
    position: u32,
    value: QI,
    scales: &Tensor<FS>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scales, position);
    dequantize_packed_value::<F, FS, QI>(value, scale, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value<F: Float, FS: Float, QS: Int>(
    value: QS,
    scale: FS,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    // TODO: q_store_type: QuantStoreType::Native
    let floats = unpack_q::<F, QS>(value, scheme.value);

    dequantize_symmetric::<F, FS>(floats, scale)
}

/// Unpack a quantized integer into a line of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, QS: Int>(value: QS, #[comptime] quant: QuantValue) -> Line<F> {
    let size_quant = comptime!(quant.size_bits() as u32);

    let size_store = comptime!(QS::size_bits().unwrap() as u32);
    let num_quant = comptime!(size_store / size_quant);

    let mut output = Line::empty(num_quant);
    let mut position = comptime!(0);

    let mask = QS::cast_from(comptime!((1 << size_quant) - 1));
    let sign_bit = QS::cast_from(comptime!(1 << (size_quant - 1)));
    let two_pow_n = comptime!(1 << size_quant);

    #[unroll]
    for _ in 0..num_quant {
        let offset = QS::cast_from(comptime!(position * size_quant));
        let raw = (value >> offset) & mask;

        // Branchless two's complement conversion
        // If raw >= 2^(n-1), then result = raw - 2^n
        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit); // 1 if negative, 0 if positive
        let signed_value = raw_i32 - (is_negative * two_pow_n);

        output[position] = F::cast_from(signed_value);
        comptime!(position += 1);
    }

    output
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_packed_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<u32>>,
    scales: &Tensor<FS>,
    output: &mut Tensor<Line<F>>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    // Input line size = 1
    let qparams = QParams::new(scheme);
    let num_quants = comptime!(qparams.num_quants);
    let scale = qparams.scale(scales, ABSOLUTE_POS);
    let value = input[ABSOLUTE_POS][0];

    let out = dequantize_packed_value::<F, FS, u32>(value, scale, scheme);

    if comptime!(output.line_size() == num_quants) {
        output[ABSOLUTE_POS] = out;
    } else {
        // Output line size = 1
        #[unroll]
        for i in 0..out.size() {
            output[ABSOLUTE_POS * out.size() + i] = Line::cast_from(out[i]);
        }
    }
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_int8_native_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<i8>>,
    scale: &Tensor<FS>,
    output: &mut Tensor<Line<F>>,
    out_layout: StridedLayout,
    #[comptime] scheme: QuantScheme,
    #[comptime] rank: Option<u32>,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    let qparams = QParams::new(scheme);
    // Absolute pos represents the logical block (scale) used to dequantize, not layout
    let scale = qparams.scale(scale, ABSOLUTE_POS * input.line_size());

    output[out_pos] = dequantize_symmetric::<F, FS>(Line::cast_from(input[in_pos]), scale);
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scale: &TensorHandleRef<'_, R>,
    scheme: &QuantScheme,
) {
    match scheme {
        QuantScheme {
            value: QuantValue::QInt8,
            store: QuantStore::U32,
            ..
        } => match scheme.param {
            QuantParam::F32 => dequantize_packed::<R, F, f32>(client, input, scheme, scale, output),
            QuantParam::F16 => dequantize_packed::<R, F, f16>(client, input, scheme, scale, output),
            QuantParam::BF16 => {
                dequantize_packed::<R, F, bf16>(client, input, scheme, scale, output)
            }
        },
        QuantScheme {
            value: QuantValue::QInt8,
            store: QuantStore::Native,
            ..
        } => {
            if !i8::is_supported(&client) {
                panic!("QInt8 is not supported for native quantization");
            }

            match scheme.param {
                QuantParam::F32 => {
                    dequantize_native::<R, F, f32>(client, input, scheme, scale, output)
                }
                QuantParam::F16 => {
                    dequantize_native::<R, F, f16>(client, input, scheme, scale, output)
                }
                QuantParam::BF16 => {
                    dequantize_native::<R, F, bf16>(client, input, scheme, scale, output)
                }
            }
        }
    }
}

fn dequantize_packed<R: Runtime, F: Float, FS: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
) {
    // The actual number of elements is 1/4 (four int8 values packed in a single u32)
    // so we choose a line size to match a valid input binding size.
    let num_out_elems: usize = input.shape.iter().product();
    let num_quants = (scheme.size_bits_stored() / scheme.size_bits_value()) as u8;
    let num_elems_input = num_out_elems / num_quants as usize;
    let line_size_in = 1;
    // let line_size = tensor_line_size_parallel(
    //     R::line_size_elem(&F::as_elem_native_unchecked()),
    //     input.shape,
    //     input.strides,
    //     input.shape.len() - 1,
    // );
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems_input / line_size_in as usize, cube_dim);

    // Output line size selected based on the number of packed values per storage type
    let num_quants = (scheme.size_bits_stored() / scheme.size_bits_value()) as u8;
    let use_packed_line_size =
        num_out_elems % num_quants as usize == 0 && R::supported_line_sizes().contains(&num_quants);

    let line_size_out = if use_packed_line_size { num_quants } else { 1 };

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            store: QuantStore::U32,
            ..
        } => {
            unsafe {
                dequantize_symmetric_packed_kernel::launch_unchecked::<F, FS, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(line_size_in),
                    scale.as_tensor_arg(1),
                    output.as_tensor_arg(line_size_out),
                    *scheme,
                )
            };
        }
        QuantScheme {
            store: QuantStore::Native,
            ..
        } => panic!("Invalid quantization storage type for scheme {scheme:?}"),
    }
}

fn dequantize_native<R: Runtime, F: Float, FS: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
) {
    let num_elems: usize = input.shape.iter().product();
    let line_size = tensor_line_size_parallel(
        R::line_size_elem(&F::as_elem_native_unchecked()),
        input.shape,
        input.strides,
        input.shape.len() - 1,
    );
    let out_layout = strided_layout(client, output);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            store: QuantStore::Native,
            ..
        } => {
            unsafe {
                dequantize_symmetric_int8_native_kernel::launch_unchecked::<F, FS, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(line_size),
                    scale.as_tensor_arg(1),
                    output.as_tensor_arg(line_size),
                    out_layout,
                    *scheme,
                    Some(input.shape.len() as u32),
                )
            };
        }
        QuantScheme {
            store: QuantStore::U32,
            ..
        } => panic!("Invalid quantization storage type for scheme {scheme:?}"),
    }
}
