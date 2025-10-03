#![allow(missing_docs)] // pub cube modules

use cubecl::prelude::*;
use cubecl_common::{e2m1x2, e4m3, e5m2, ue8m0};
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};
use cubecl_runtime::TypeUsage;

use crate::{
    layout::{ScalesView, scales_view},
    scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubecl_std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};
use half::{bf16, f16};

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: CubePrimitive>(value: Line<F>, scale: FS) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * value
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_values<F: Float, FS: CubePrimitive, QI: Int>(
    position: u32,
    values: &View<Line<QI>, u32>,
    scales: &View<FS, u32>,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    dequantize_symmetric_packed_value_at::<F, FS, QI>(position, values[position], scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value_at<F: Float, FS: CubePrimitive, QI: Int>(
    position: u32,
    values: Line<QI>,
    scales: &View<FS, u32>,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    dequantize_symmetric_packed_value::<F, FS, QI>(values, scales, position, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value<F: Float, FS: CubePrimitive, QS: Int>(
    values: Line<QS>,
    scales: &View<FS, u32>,
    position: u32,
    #[comptime] scheme: QuantScheme,
) -> Array<Line<F>> {
    let line_size_values = values.line_size();
    let num_quants = comptime!(scheme.num_quants() as u32);
    let mut tmp = Array::vectorized(line_size_values, num_quants);

    #[unroll]
    for i in 0..line_size_values {
        let floats = unpack_q::<F, QS>(values[i], scheme.value, scheme.store);
        let scale = scales[(position * line_size_values) + i * num_quants];
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
fn dequantize_symmetric_packed_kernel<F: Float, FS: CubePrimitive>(
    input: &LinearView<Line<u32>>,
    scales: &ScalesView<FS>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    #[comptime] scheme: QuantScheme,
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let line_size_in = input.line_size();
    let line_size_out = output.line_size();

    comptime! {
        assert_eq!(line_size_out, scheme.num_quants() as u32);
    }

    let values = input[ABSOLUTE_POS];
    let packed_pos = ABSOLUTE_POS * comptime![scheme.num_quants() as u32];

    let out = dequantize_symmetric_packed_value::<F, FS, u32>(values, scales, packed_pos, scheme);

    #[unroll]
    for i in 0..line_size_in {
        output[ABSOLUTE_POS * line_size_in + i] = out[i];
    }
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_native_kernel<F: Float, FS: CubePrimitive, Q: CubePrimitive>(
    input: &LinearView<Line<Q>>,
    scale: &ScalesView<FS>,
    output: &mut LinearView<Line<F>, ReadWrite>,
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    // Absolute pos represents the logical block (scale) used to dequantize, not layout
    let scale = scale[ABSOLUTE_POS * input.line_size() * native_packing];

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
                dequantize_packed::<R, F, f32>(client, values, *scheme, params, output)
            }
            QuantParam::F16 => {
                dequantize_packed::<R, F, f16>(client, values, *scheme, params, output)
            }
            QuantParam::BF16 => {
                dequantize_packed::<R, F, bf16>(client, values, *scheme, params, output)
            }
            QuantParam::UE8M0 => {
                dequantize_packed::<R, F, ue8m0>(client, values, *scheme, params, output)
            }
            QuantParam::UE4M3 => {
                dequantize_packed::<R, F, e4m3>(client, values, *scheme, params, output)
            }
        },
        QuantScheme {
            value:
                QuantValue::Q8F
                | QuantValue::Q8S
                | QuantValue::E4M3
                | QuantValue::E5M2
                | QuantValue::E2M1,
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
                    dequantize_native::<R, F, f32>(client, values, *scheme, params, output)
                }
                QuantParam::F16 => {
                    dequantize_native::<R, F, f16>(client, values, *scheme, params, output)
                }
                QuantParam::BF16 => {
                    dequantize_native::<R, F, bf16>(client, values, *scheme, params, output)
                }
                QuantParam::UE8M0 => {
                    dequantize_native::<R, F, ue8m0>(client, values, *scheme, params, output)
                }
                QuantParam::UE4M3 => {
                    dequantize_native::<R, F, e4m3>(client, values, *scheme, params, output)
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

fn dequantize_packed<R: Runtime, F: Float, FS: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
) {
    let num_elems_input: usize = input.shape.iter().product();

    let mut line_size_in = tensor_line_size_parallel(
        R::io_optimized_line_sizes_unchecked(&F::as_type_native_unchecked()),
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
                    linear_view(client, input, line_size_in),
                    scales_view(client, input, scale, 1, &scheme),
                    linear_view(client, output, line_size_out),
                    scheme,
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}

fn dequantize_native<R: Runtime, F: Float, FS: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
) {
    let num_elems: usize = input.shape.iter().product();
    let line_size = tensor_line_size_parallel(
        R::io_optimized_line_sizes_unchecked(&F::as_type_native_unchecked()),
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
            value,
            store: QuantStore::Native,
            ..
        } => {
            let launch = match value {
                QuantValue::Q8F | QuantValue::Q8S => {
                    dequantize_symmetric_native_kernel::launch_unchecked::<F, FS, i8, R>
                }
                QuantValue::E4M3 => {
                    dequantize_symmetric_native_kernel::launch_unchecked::<F, FS, e4m3, R>
                }
                QuantValue::E5M2 => {
                    dequantize_symmetric_native_kernel::launch_unchecked::<F, FS, e5m2, R>
                }
                QuantValue::E2M1 => {
                    dequantize_symmetric_native_kernel::launch_unchecked::<F, FS, e2m1x2, R>
                }
                other => panic!("Unsupported quantization value {other:?}"),
            };

            unsafe {
                launch(
                    client,
                    cube_count,
                    cube_dim,
                    linear_view(client, input, line_size),
                    scales_view(client, input, scale, 1, &scheme),
                    linear_view(client, output, line_size),
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}
