use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::tensor_line_size_parallel;
use cubecl_std::tensor::into_contiguous;
use cubecl_std::tensor::{StridedLayout, index_offset_contiguous};

use crate::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};
use crate::utils::check_block_size_compat;
use crate::utils::strided_layout;
use half::{bf16, f16};

#[cube]
fn quantize_symmetric<F: Float, FS: Float>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
) -> Line<F> {
    Line::clamp(
        Line::round(value / Line::cast_from(scale)),
        Line::new(range_min),
        Line::new(range_max),
    )
}

#[cube]
fn quantize_symmetric_i<F: Float, FS: Float, I: Int>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
) -> Line<I> {
    Line::cast_from(quantize_symmetric::<F, FS>(
        value, scale, range_min, range_max,
    ))
}

#[cube]
fn quantize_packed_value<F: Float, FS: Float, QS: Int>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
    #[comptime] scheme: QuantScheme,
) -> QS {
    let value = quantize_symmetric::<F, FS>(value, scale, range_min, range_max);
    pack_q::<F, QS>(value, scheme.value)
}

/// Pack a line of quantized floating-point values into a single integer (the stored quantization type),
/// according to the specified quantization input type.
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn pack_q<F: Float, QS: Int>(value: Line<F>, #[comptime] quant: QuantValue) -> QS {
    let size_quant = comptime!(quant.size_bits() as u32);

    let size_store = comptime!(QS::size_bits().unwrap() as u32);
    let num_quants = comptime!(size_store / size_quant);

    let mask = i32::cast_from(comptime!((1 << size_quant) - 1));
    let mut position = comptime!(0);
    let mut packed = QS::cast_from(0);

    // Shift and combine into QS (using i32 for sign extension)
    #[unroll]
    for _ in 0..num_quants {
        let offset = QS::cast_from(comptime!(position * size_quant));
        let shifted = QS::cast_from(i32::cast_from(value[position]) & mask) << offset;
        packed |= shifted;
        comptime!(position += 1);
    }

    packed
}

#[cube]
fn write_scale_per_tensor<F: Float, FS: Float>(
    in_pos: u32,
    scale: &Tensor<F>,
    out_scale: &mut Tensor<FS>,
) -> FS {
    let scale = FS::cast_from(scale[0]);

    // Write the scale into the output buffer
    if in_pos == 0 {
        out_scale[in_pos] = scale;
    }

    scale
}

#[cube]
fn write_scale_per_block<F: Float, FS: Float>(
    in_pos: u32,
    scale: &Tensor<F>,
    out_scale: &mut Tensor<FS>,
    #[comptime] block_size: u32,
) -> FS {
    let scale_pos = in_pos / block_size;
    let scale = FS::cast_from(scale[scale_pos]);

    // Write the scale into the output buffer
    if in_pos % block_size == 0 {
        out_scale[scale_pos] = scale;
    }

    scale
}

#[cube(launch_unchecked)]
fn quantize_symmetric_int8_native_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<F>>,
    scale: &Tensor<F>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<Line<i8>>,
    out_scale: &mut Tensor<FS>,
    out_layout: StridedLayout,
    #[comptime] rank: Option<u32>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    let scale = match comptime!(scheme) {
        QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } => write_scale_per_block(
            in_pos * input.line_size(),
            scale,
            out_scale,
            comptime!(block_size as u32),
        ),
        QuantScheme {
            level: QuantLevel::Tensor,
            ..
        } => write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale),
    };

    output[out_pos] = quantize_symmetric_i::<F, FS, i8>(input[in_pos], scale, range_min, range_max);
}

#[cube(launch_unchecked)]
fn quantize_symmetric_int8_packed_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<F>>,
    scale: &Tensor<F>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<u32>,
    out_scale: &mut Tensor<FS>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let num_quants = comptime!(scheme.num_quants() as u32);
    let packed_pos = ABSOLUTE_POS * num_quants;

    let scale = match comptime!(scheme) {
        QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } => write_scale_per_block(packed_pos, scale, out_scale, comptime!(block_size as u32)),
        QuantScheme {
            level: QuantLevel::Tensor,
            ..
        } => write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale),
    };

    if comptime!(input.line_size() == num_quants) {
        output[ABSOLUTE_POS] = quantize_packed_value::<F, FS, u32>(
            input[ABSOLUTE_POS],
            scale,
            range_min,
            range_max,
            scheme,
        );
    } else {
        // Input line size = 1
        let mut values = Line::<F>::empty(num_quants);
        #[unroll]
        for i in 0..num_quants {
            values[i] = input[packed_pos + i][0];
        }
        output[ABSOLUTE_POS] =
            quantize_packed_value::<F, FS, u32>(values, scale, range_min, range_max, scheme);
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
    scheme: &QuantScheme,
) {
    match scheme {
        QuantScheme {
            store: QuantStore::U32,
            ..
        } => match scheme.param {
            QuantParam::F32 => {
                quantize_packed::<R, F, f32>(client, input, scheme, scale, out_scale, output)
            }
            QuantParam::F16 => {
                quantize_packed::<R, F, f16>(client, input, scheme, scale, out_scale, output)
            }
            QuantParam::BF16 => {
                quantize_packed::<R, F, bf16>(client, input, scheme, scale, out_scale, output)
            }
        },
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S,
            store: QuantStore::Native,
            ..
        } => {
            if !i8::is_supported(client) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            match scheme.param {
                QuantParam::F32 => {
                    quantize_native::<R, F, f32>(client, input, scheme, scale, out_scale, output)
                }
                QuantParam::F16 => {
                    quantize_native::<R, F, f16>(client, input, scheme, scale, out_scale, output)
                }
                QuantParam::BF16 => {
                    quantize_native::<R, F, bf16>(client, input, scheme, scale, out_scale, output)
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

fn quantize_native<R: Runtime, F: Float, FS: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
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
    let (range_min, range_max) = scheme.value.range();

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value: QuantValue::Q8S,
            store: QuantStore::Native,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            check_block_size_compat(scheme, line_size as usize);
            unsafe {
                quantize_symmetric_int8_native_kernel::launch_unchecked::<F, FS, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scale.as_tensor_arg(1),
                    ScalarArg::new(F::from_int(range_min as i64)),
                    ScalarArg::new(F::from_int(range_max as i64)),
                    output.as_tensor_arg(line_size),
                    out_scale.as_tensor_arg(1),
                    out_layout,
                    Some(input.shape.len() as u32),
                    *scheme,
                )
            };
        }
        _ => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}

fn quantize_packed<R: Runtime, F: Float, FS: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
) {
    let input = into_contiguous::<R, F>(client, input);
    let num_elems: usize = output.shape.iter().product();

    let num_quants = scheme.num_quants() as u8;
    let line_size = num_quants;

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(line_size as usize), cube_dim);
    let (range_min, range_max) = scheme.value.range();

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            store: QuantStore::U32,
            ..
        } => {
            check_block_size_compat(scheme, num_quants as usize); // 32 / 8 = 4
            unsafe {
                quantize_symmetric_int8_packed_kernel::launch_unchecked::<F, FS, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_ref().as_tensor_arg(line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scale.as_tensor_arg(1),
                    ScalarArg::new(F::from_int(range_min as i64)),
                    ScalarArg::new(F::from_int(range_max as i64)),
                    output.as_tensor_arg(1),
                    out_scale.as_tensor_arg(1),
                    *scheme,
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}
