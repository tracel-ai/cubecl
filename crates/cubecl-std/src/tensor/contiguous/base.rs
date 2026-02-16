use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        TensorHandle, into_contiguous_ref,
        layout::{
            Layout, LayoutExpand,
            linear::{LinearLayout, LinearLayoutArgs, LinearView, linear_view},
        },
    },
};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise,
    ir::{LineSize, StorageType},
    tensor_line_size_parallel,
};

pub const SM_COUNT_APPROX: u32 = 50;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    layout: &Tensor<Line<L>>,
    offset_layout: usize,
    axis_start: usize,
    axis_end: usize,
    #[comptime] unroll: bool,
) -> usize {
    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;

    #[unroll(unroll)]
    for i in axis_start..axis_end {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous<N: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    offset_layout: usize,
    #[comptime] rank: Option<usize>,
) -> usize {
    let unroll = rank.is_some();
    let rank = rank.unwrap_or_else(|| tensor.rank());

    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll(unroll)]
    for i in 0..rank {
        let axis = rank - i - 1;
        let shape = tensor.shape(axis);
        let ogwl = remainder % shape;
        offset += ogwl * tensor.stride(axis);
        remainder /= shape;
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous_fastdivmod(
    offset: usize,
    shape: &Sequence<FastDivmod<usize>>,
    stride: &Sequence<usize>,
    #[comptime] line_size: LineSize,
) -> usize {
    let rank = shape.len().comptime();

    let offset_ref = offset * line_size;
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll]
    for i in 0..rank {
        let axis = rank - i - 1;

        let (rem, ogwl) = shape[axis].div_mod(remainder);
        offset += ogwl * stride[axis];
        remainder = rem;
    }

    offset / line_size
}

#[cube(launch, address_type = "dynamic")]
fn copy_kernel<N: Numeric>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let offset_linear = ABSOLUTE_POS * elems_per_thread;
    let line_size = input.line_size();

    let mut registers = Array::<Line<N>>::lined(elems_per_thread, line_size);

    #[unroll]
    for i in 0..elems_per_thread {
        registers[i] = input[offset_linear + i];
    }

    let offset_output = out_layout.to_source_pos(offset_linear);

    #[unroll]
    for i in 0..elems_per_thread {
        output[offset_output + i] = registers[i];
    }
}

#[cube(launch, address_type = "dynamic")]
fn copy_kernel_pack<N: Numeric>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let line_size = output.line_size().comptime();
    let lines_per_thread = elems_per_thread / line_size;

    let offset_output = ABSOLUTE_POS * lines_per_thread;
    let offset_input = offset_output * line_size;

    let mut registers = Array::<Line<N>>::lined(lines_per_thread, line_size);

    #[unroll]
    for i in 0..lines_per_thread {
        let offset = i * line_size;
        let mut reg = Line::<N>::empty(line_size);
        #[unroll]
        for k in 0..line_size {
            let offset_input = offset_input + offset + k;
            reg[k] = input[offset_input][0];
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..lines_per_thread {
        output[offset_output + i] = registers[i];
    }
}

/// Fetch all values required contained in a given position, unpack them, then repack them to their
/// new position.
#[cube]
fn index_packed<N: Int>(
    tensor: &Tensor<N>,
    pos: usize,
    in_shape: &Sequence<FastDivmod<usize>>,
    #[comptime] packed_axis: usize,
    #[comptime] packing: usize,
    #[comptime] rank: usize,
) -> N {
    let type_size_bits = N::type_size_bits().comptime();
    let bits_per_elem = type_size_bits / packing;
    let mask = (1u32 << bits_per_elem) - 1;
    let mask = N::cast_from(mask);

    let elem_pos = pos * packing;

    let mut out = N::new(0);
    for n in 0..packing {
        let mut remainder = elem_pos + n;
        let mut offset = 0;
        let mut packing_offset = 0;

        #[unroll]
        for i in 0..rank {
            let axis = rank - i - 1;
            let (rem, mut local_pos) = in_shape[axis].div_mod(remainder);
            remainder = rem;
            if axis == packed_axis {
                packing_offset = local_pos % packing;
                local_pos /= packing;
            }
            offset += local_pos * tensor.stride(axis);
        }
        let packed_val = tensor[offset];
        let shift_in = packing_offset * bits_per_elem;
        let shift_out = n * bits_per_elem;
        let value = (packed_val >> N::cast_from(shift_in)) & mask;

        out |= value << N::cast_from(shift_out);
    }
    out
}

#[cube(launch, address_type = "dynamic")]
fn copy_kernel_packed<N: Int>(
    input: &Tensor<N>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] packed_axis: usize,
    #[comptime] packing: usize,
    #[comptime] rank: usize,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let line_size = output.line_size().comptime();
    let lines_per_thread = elems_per_thread / line_size;

    let offset_output = ABSOLUTE_POS * lines_per_thread;
    let offset_input = offset_output * line_size;

    if offset_output >= output.len() {
        terminate!()
    }

    let mut registers = Array::<Line<N>>::lined(lines_per_thread, line_size);

    #[unroll]
    for i in 0..lines_per_thread {
        let offset = i * line_size;
        let mut reg = Line::<N>::empty(line_size);
        #[unroll]
        for k in 0..line_size {
            let offset_input = offset_input + offset + k;

            reg[k] = index_packed(input, offset_input, &in_shape, packed_axis, packing, rank);
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..lines_per_thread {
        output[offset_output + i] = registers[i];
    }
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [`create_tensor`](cubecl_runtime::client::ComputeClient::create_tensor).
/// Handles unpacking and repacking packed tensors (i.e. quantized values).
/// `shape` refers to the actual (unpacked) shape of the tensor, while `packing` specifies the
/// number of elements in each storage element.
///
/// # Warning
/// This assumes `u32` or `u8` packing.
pub fn into_contiguous_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    packed_axis: usize,
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let rank = shape.len();
    if rank <= 1 {
        return into_contiguous_ref(client, input, dtype);
    }

    let mut out_shape = shape.to_vec();
    out_shape[rank - 1] = out_shape[rank - 1].div_ceil(packing);
    let output = TensorHandle::empty(client, out_shape, dtype);

    // Should reinterpret as u8 if possible at some point, but requires modifying shape/strides so
    // keep it simple for now
    into_contiguous_packed_ref(
        client,
        input,
        &output.as_ref(),
        packed_axis,
        shape,
        packing,
        dtype,
    )?;

    Ok(output)
}

/// Make a jit tensor contiguous.
pub fn copy_gpu_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let elem_count: usize = input.shape.iter().product();

    // Line size is only enabled when the last dimension is contiguous.
    let in_rank = input.strides.len();
    let out_rank = output.strides.len();
    let line_size_in = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtype),
        input.shape,
        input.strides,
        in_rank - 1,
    );
    let line_size_out = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtype),
        output.shape,
        output.strides,
        out_rank - 1,
    );
    let line_size = line_size_in.min(line_size_out);

    let vec_count = elem_count / line_size as usize;
    let sm_count = client
        .properties()
        .hardware
        .streaming_multiprocessor_count
        .unwrap_or(SM_COUNT_APPROX);
    let cube_dim = CubeDim::new(client, vec_count);
    let simul_vecs = sm_count * cube_dim.num_elems();
    let mut elems_per_unit = match vec_count / simul_vecs as usize {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut elems_per_unit_count = line_size as usize * elems_per_unit;

    let last_axis = output.shape[out_rank - 1];

    // If tensor is strided, elems_per_unit must be compatible with last axis
    while !last_axis.is_multiple_of(elems_per_unit_count as usize) {
        elems_per_unit /= 2;
        elems_per_unit_count /= 2;
    }

    let out_vec = if line_size > 1 {
        line_size
    } else {
        // Recompute because it needs to account for `elems_per_unit_count`
        client
            .io_optimized_line_sizes(&dtype)
            .filter(|it| elems_per_unit_count.is_multiple_of(*it))
            .max()
            .unwrap_or(1)
    };

    let address_type = input
        .required_address_type()
        .max(output.required_address_type());
    let input = linear_view(client, input, line_size);
    let out_layout = LinearLayoutArgs::from_handle(client, output, out_vec);

    let cube_count = calculate_cube_count_elemwise(
        client,
        elem_count.div_ceil(elems_per_unit_count as usize),
        cube_dim,
    );

    let launch = if line_size != out_vec && out_vec > 1 {
        copy_kernel_pack::launch
    } else {
        copy_kernel::launch
    };

    launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        input,
        output.as_tensor_arg(out_vec),
        out_layout,
        elems_per_unit,
        dtype,
    )
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_packed_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    packed_axis: usize,
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let elem_count: usize = input.shape.iter().product();

    // Line size is only enabled when the last dimension is contiguous.
    let in_rank = input.strides.len();
    let out_rank = output.strides.len();
    let in_packed_axis = in_rank - packed_axis - 1;
    let line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtype),
        output.shape,
        output.strides,
        out_rank - 1,
    );
    let vec_count = elem_count / line_size as usize;
    let sm_count = client
        .properties()
        .hardware
        .streaming_multiprocessor_count
        .unwrap_or(SM_COUNT_APPROX);

    let cube_dim = CubeDim::new(client, vec_count);
    let simul_vecs = sm_count * cube_dim.num_elems();
    let mut elems_per_unit = match vec_count / simul_vecs as usize {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut elems_per_unit_count = line_size as usize * elems_per_unit;

    let last_axis = output.shape[out_rank - 1];

    // If tensor is strided, elems_per_unit must be compatible with last axis
    while !last_axis.is_multiple_of(elems_per_unit_count as usize) {
        elems_per_unit /= 2;
        elems_per_unit_count /= 2;
    }

    let out_layout = LinearLayoutArgs::from_handle(client, output, line_size);

    let address_type = input
        .required_address_type()
        .max(output.required_address_type());
    let cube_count = calculate_cube_count_elemwise(
        client,
        elem_count.div_ceil(elems_per_unit_count as usize),
        cube_dim,
    );

    let in_shape = shape
        .iter()
        .map(|s| FastDivmodArgs::<usize>::new(client, *s))
        .collect();

    copy_kernel_packed::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        input.as_tensor_arg(1),
        output.as_tensor_arg(line_size),
        out_layout,
        in_shape,
        in_packed_axis,
        packing,
        in_rank,
        elems_per_unit,
        dtype,
    )
}

/// Checks if the tensor associated with the given shape and strides is contiguous.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    for (expected, &stride) in compact_strides(shape).into_iter().zip(strides) {
        if expected != stride {
            return false;
        }
    }

    true
}

/// Checks if a tensor is only strided on the last dimension, and could be safely reinterpreted as
/// a 2D tensor with unit stride on the last dimension. This will always hold for non-permuted
/// tensors allocated on a runtime.
pub fn is_contiguous_pitched(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();
    if strides[rank - 1] != 1 {
        return false;
    }
    if rank <= 1 {
        return true;
    }

    let mut sorted = strides.to_vec();
    sorted.sort();
    sorted.reverse();

    if sorted != strides {
        return false;
    }

    for i in 0..rank - 2 {
        if strides[i] != shape[i + 1] * strides[i + 1] {
            return false;
        }
    }
    true
}

pub fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
