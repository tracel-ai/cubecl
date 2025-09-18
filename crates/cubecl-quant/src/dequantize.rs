#![allow(missing_docs)] // pub cube modules

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};
use cubecl_runtime::TypeUsage;

use crate::{
    qparams::QParams,
    scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubecl_std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};
use half::{bf16, f16};

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: Float>(value: Line<F>, scale: FS) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * value
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_values<F: Float, FS: Float, QI: Int>(
    position: u32,
    values: &View<Line<QI>, u32>,
    scales: &View<Line<FS>, u32>,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    dequantize_symmetric_packed_value_at::<F, FS, QI>(position, values[position], scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value_at<F: Float, FS: Float, QI: Int>(
    position: u32,
    values: Line<QI>,
    scales: &View<Line<FS>, u32>,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    let qparams = QParams::new(scheme);
    dequantize_symmetric_packed_value::<F, FS, QI>(values, scales, qparams, position, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value<F: Float, FS: Float, QS: Int>(
    values: Line<QS>,
    scales: &View<Line<FS>, u32>,
    qparams: QParams,
    position: u32,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    let line_size_values = values.line_size();
    let num_quants = comptime!(qparams.num_quants);
    let mut tmp = Array::vectorized(line_size_values, num_quants);

    #[unroll]
    for i in 0..line_size_values {
        let floats = unpack_q::<F, QS>(values[i], scheme.value, scheme.store);
        let scale = qparams.scale(scales, (position * line_size_values) + i);
        let values = dequantize_symmetric::<F, FS>(floats, scale);
        tmp[i] = values;
    }

    tmp
}

/// Unpack a quantized integer into a line of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, QS: Int>(
    value: QS,
    #[comptime] quant: QuantValue,
    #[comptime] store: QuantStore,
) -> Line<F> {
    let size_quant = comptime!(quant.size_bits() as u32);
    let size_store = comptime!(store.size_bits(&quant) as u32);
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
    input: &LinearView<Line<u32>>,
    scales: &LinearView<Line<FS>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    #[comptime] scheme: QuantScheme,
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    let line_size_in = input.line_size();
    let line_size_out = output.line_size();

    comptime! {
        assert_eq!(line_size_out, qparams.num_quants);
    }

    let values = input[ABSOLUTE_POS];

    let out = dequantize_symmetric_packed_value::<F, FS, u32>(
        values,
        scales,
        qparams,
        ABSOLUTE_POS,
        scheme,
    );

    #[unroll]
    for i in 0..line_size_in {
        output[ABSOLUTE_POS * line_size_in + i] = out[i];
    }
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_int8_native_kernel<F: Float, FS: Float>(
    input: &LinearView<Line<i8>>,
    scale: &LinearView<Line<FS>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    #[comptime] scheme: QuantScheme,
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    // Absolute pos represents the logical block (scale) used to dequantize, not layout
    let scale = qparams.scale(scale, ABSOLUTE_POS * input.line_size());

    output[ABSOLUTE_POS] =
        dequantize_symmetric::<F, FS>(Line::cast_from(input[ABSOLUTE_POS]), scale);
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    values: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    params: &TensorHandleRef<'_, R>,
    scheme: &QuantScheme,
) {
    match scheme {
        QuantScheme {
            store: QuantStore::U32,
            ..
        } => match scheme.param {
            QuantParam::F32 => {
                dequantize_packed::<R, F, f32>(client, values, scheme, params, output)
            }
            QuantParam::F16 => {
                dequantize_packed::<R, F, f16>(client, values, scheme, params, output)
            }
            QuantParam::BF16 => {
                dequantize_packed::<R, F, bf16>(client, values, scheme, params, output)
            }
        },
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S,
            store: QuantStore::Native,
            ..
        } => {
            if !i8::supported_uses(client).contains(TypeUsage::Conversion) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            match scheme.param {
                QuantParam::F32 => {
                    dequantize_native::<R, F, f32>(client, values, scheme, params, output)
                }
                QuantParam::F16 => {
                    dequantize_native::<R, F, f16>(client, values, scheme, params, output)
                }
                QuantParam::BF16 => {
                    dequantize_native::<R, F, bf16>(client, values, scheme, params, output)
                }
            }
        }
        QuantScheme {
            store: QuantStore::Native,
            value,
            ..
        } => {
            panic!("{value:?} is not supported for native quantization");
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
    let num_elems_input: usize = input.shape.iter().product();

    let mut line_size_in = tensor_line_size_parallel(
        R::line_size_type(&F::as_type_native_unchecked()),
        input.shape,
        input.strides,
        input.shape.len() - 1,
    );
    let num_quants = scheme.num_quants() as u8;
    let line_size_out = num_quants;
    let rank = output.shape.len();

    if !output.shape[rank - 1].is_multiple_of(line_size_out as usize) {
        line_size_in = 1;
    }

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems_input / line_size_in as usize, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            store: QuantStore::U32,
            mode: QuantMode::Symmetric,
            ..
        } => {
            unsafe {
                dequantize_symmetric_packed_kernel::launch_unchecked::<F, FS, R>(
                    client,
                    cube_count,
                    cube_dim,
                    linear_view(client, input, &line_size_in),
                    linear_view(client, scale, &1),
                    linear_view(client, output, &line_size_out),
                    *scheme,
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
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
        R::line_size_type(&F::as_type_native_unchecked()),
        input.shape,
        input.strides,
        input.shape.len() - 1,
    );
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::Q8F | QuantValue::Q8S,
            store: QuantStore::Native,
            ..
        } => {
            unsafe {
                dequantize_symmetric_int8_native_kernel::launch_unchecked::<F, FS, R>(
                    client,
                    cube_count,
                    cube_dim,
                    linear_view(client, input, &line_size),
                    linear_view(client, scale, &1),
                    linear_view(client, output, &line_size),
                    *scheme,
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}
