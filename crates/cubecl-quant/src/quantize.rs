use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_core::ir::ElemType;
use cubecl_core::tensor_line_size_parallel;
use cubecl_core::{self as cubecl};
use cubecl_runtime::TypeUsage;
use cubecl_std::scalar::InputScalar;
use cubecl_std::tensor::layout::linear::LinearView;
use cubecl_std::tensor::{View, layout::linear::linear_view};

use crate::{
    layout::{ScalesLayout, scales_view},
    utils::check_block_size_compat,
};
use crate::{
    layout::{ScalesView, scales_layout},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
};

#[cube]
fn quantize_symmetric<F: Float, FS: CubePrimitive>(
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
fn quantize_symmetric_q<F: Float, FS: CubePrimitive, Q: CubePrimitive>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
) -> Line<Q> {
    Line::cast_from(quantize_symmetric::<F, FS>(
        value, scale, range_min, range_max,
    ))
}

#[cube]
fn quantize_packed_value<F: Float, FS: CubePrimitive, QS: Int>(
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
fn write_scale<F: Float, FS: CubePrimitive>(
    in_pos: u32,
    scale: &View<F, u32>,
    out_scale: &mut View<FS, u32, ReadWrite>,
    scales_layout: ScalesLayout,
) -> FS {
    let scale = FS::cast_from(scale[in_pos]);

    // Write the scale into the output buffer
    if scales_layout.is_block_start(in_pos) {
        out_scale[in_pos] = scale;
    }

    scale
}

#[cube(launch_unchecked)]
fn quantize_symmetric_native_kernel<F: Float, FS: Numeric, Q: Numeric>(
    input: &LinearView<Line<F>>,
    scale: &ScalesView<F>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<Line<Q>, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    #[define(F, FS, Q)] _dtypes: [StorageType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let in_pos = ABSOLUTE_POS * input.line_size() * native_packing;
    let scale = write_scale(in_pos, scale, out_scale, scales_layout);

    output[ABSOLUTE_POS] = quantize_symmetric_q::<F, FS, Q>(
        input[ABSOLUTE_POS],
        scale,
        range_min.get::<F>(),
        range_max.get::<F>(),
    );
    sync_cube();
}

#[cube(launch_unchecked)]
fn quantize_symmetric_packed_kernel<F: Float, FS: Numeric>(
    input: &LinearView<Line<F>>,
    scale: &ScalesView<F>,
    range_min: InputScalar,
    range_max: InputScalar,
    output: &mut LinearView<Line<u32>, ReadWrite>,
    out_scale: &mut ScalesView<FS, ReadWrite>,
    scales_layout: ScalesLayout,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let num_quants = comptime!(scheme.num_quants() as u32);
    let packed_pos = ABSOLUTE_POS * num_quants;
    let scale = write_scale(packed_pos, scale, out_scale, scales_layout);

    if comptime!(input.line_size() == num_quants) {
        output[ABSOLUTE_POS] = Line::cast_from(quantize_packed_value::<F, FS, u32>(
            input[ABSOLUTE_POS],
            scale,
            range_min.get::<F>(),
            range_max.get::<F>(),
            scheme,
        ));
    } else {
        // Input line size = 1
        let mut values = Line::<F>::empty(num_quants);
        #[unroll]
        for i in 0..num_quants {
            values[i] = input[packed_pos + i][0];
        }
        output[ABSOLUTE_POS] = Line::cast_from(quantize_packed_value::<F, FS, u32>(
            values,
            scale,
            range_min.get::<F>(),
            range_max.get::<F>(),
            scheme,
        ));
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
    scheme: &QuantScheme,
    input_elem: ElemType,
) {
    let param_elem = ElemType::from_quant_param(scheme.param);

    match scheme {
        QuantScheme {
            store: QuantStore::U32,
            ..
        } => quantize_packed::<R>(
            client, input, scheme, scale, out_scale, output, input_elem, param_elem,
        ),
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

            quantize_native::<R>(
                client, input, scheme, scale, out_scale, output, input_elem, param_elem,
            )
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

#[allow(clippy::too_many_arguments)]
fn quantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
    input_dtype: ElemType,
    scale_dtype: ElemType,
) {
    let num_elems: usize = input.shape.iter().product();
    let line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes_unchecked(input.elem_size),
        input.shape,
        input.strides,
        input.shape.len() - 1,
    );
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);
    let (range_min, range_max) = scheme.value.range();

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            store: QuantStore::Native,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            check_block_size_compat(scheme, line_size as usize);
            let quant_type = ElemType::from_quant_value(scheme.value);

            unsafe {
                quantize_symmetric_native_kernel::launch_unchecked::<R>(
                    client,
                    cube_count,
                    cube_dim,
                    linear_view(client, input, line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scales_view(client, output, scale, 1, scheme),
                    InputScalar::new(range_min, input_dtype),
                    InputScalar::new(range_max, input_dtype),
                    linear_view(client, output, line_size),
                    scales_view(client, output, out_scale, 1, scheme),
                    scales_layout(client, output, scale, 1, scheme),
                    [input_dtype.into(), scale_dtype.into(), quant_type.into()],
                )
            };
        }
        _ => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    scheme: &QuantScheme,
    scale: &TensorHandleRef<'_, R>,
    out_scale: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<R>,
    dtype_input: ElemType,
    dtype_param: ElemType,
) {
    let num_elems: usize = input.shape.iter().product();

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
                quantize_symmetric_packed_kernel::launch_unchecked::<R>(
                    client,
                    cube_count,
                    cube_dim,
                    linear_view(client, input, line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scales_view(client, output, scale, 1, scheme),
                    InputScalar::new(range_min, dtype_input),
                    InputScalar::new(range_max, dtype_input),
                    linear_view(client, output, 1),
                    scales_view(client, output, out_scale, 1, scheme),
                    scales_layout(client, output, scale, 1, scheme),
                    *scheme,
                    [dtype_input.into(), dtype_param.into()],
                )
            };
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    }
}
