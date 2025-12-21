use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{
        Layout, LayoutExpand,
        linear::{LinearLayout, LinearLayoutArgs, LinearView, linear_view},
    },
};

use super::TensorHandle;
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise,
    ir::{LineSize, StorageType},
    tensor_line_size_parallel,
};

pub const NUM_SM_APPROX: u32 = 50;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    layout: &Tensor<Line<L>>,
    offset_layout: usize,
    dim_start: u32,
    dim_end: u32,
    #[comptime] unroll: bool,
) -> usize {
    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;

    #[unroll(unroll)]
    for i in dim_start..dim_end {
        let ogwl = offset_ref / layout.stride(i as usize);
        offset += ogwl % tensor.shape(i as usize) * tensor.stride(i as usize);
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous<N: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    offset_layout: usize,
    #[comptime] rank: Option<u32>,
) -> usize {
    let unroll = rank.is_some();
    let rank = rank.unwrap_or_else(|| tensor.rank() as u32);

    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll(unroll)]
    for i in 0..rank {
        let dim = rank - i - 1;
        let shape = tensor.shape(dim as usize);
        let ogwl = remainder % shape;
        offset += ogwl * tensor.stride(dim as usize);
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
    let rank = comptime![shape.len()];

    let offset_ref = offset * line_size;
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];

        let (rem, ogwl) = shape[dim].div_mod(remainder);
        offset += ogwl * stride[dim];
        remainder = rem;
    }

    offset / line_size
}

#[cube(launch)]
fn into_contiguous_kernel<N: Numeric>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let offset_output = ABSOLUTE_POS * elems_per_thread;
    let line_size = input.line_size();

    let mut registers = Array::<Line<N>>::lined(elems_per_thread, line_size);

    #[unroll]
    for i in 0..elems_per_thread {
        registers[i] = input[offset_output + i];
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..elems_per_thread {
        output[offset_output + i] = registers[i];
    }
}

#[cube(launch)]
fn into_contiguous_kernel_pack<N: Numeric>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let line_size = output.line_size();
    let lines_per_thread = comptime![elems_per_thread / line_size];

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
    #[comptime] packed_dim: usize,
    #[comptime] packing: usize,
    #[comptime] rank: usize,
) -> N {
    let type_size_bits = N::type_size_bits();
    let bits_per_elem = comptime![type_size_bits / packing];
    let mask = comptime![(1u32 << bits_per_elem) - 1];
    let mask = N::cast_from(mask);

    let elem_pos = pos * packing;

    let mut out = N::new(0);
    for n in 0..packing {
        let mut remainder = elem_pos + n;
        let mut offset = 0;
        let mut packing_offset = 0;

        #[unroll]
        for i in 0..rank {
            let dim = comptime![rank - i - 1];
            let (rem, mut local_pos) = in_shape.index(dim).div_mod(remainder);
            remainder = rem;
            if comptime![dim == packed_dim] {
                packing_offset = local_pos % packing;
                local_pos /= packing;
            }
            offset += local_pos * tensor.stride(dim);
        }
        let packed_val = tensor[offset];
        let shift_in = packing_offset * bits_per_elem;
        let shift_out = n * bits_per_elem;
        let value = (packed_val >> N::cast_from(shift_in)) & mask;

        out |= value << N::cast_from(shift_out);
    }
    out
}

#[cube(launch)]
fn into_contiguous_kernel_packed<N: Int>(
    input: &Tensor<N>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] packed_dim: usize,
    #[comptime] packing: usize,
    #[comptime] rank: usize,
    #[comptime] elems_per_thread: usize,
    #[define(N)] _elem: StorageType,
) {
    let line_size = output.line_size();
    let lines_per_thread = comptime![elems_per_thread / line_size];

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

            reg[k] = index_packed(input, offset_input, &in_shape, packed_dim, packing, rank);
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..lines_per_thread {
        output[offset_output + i] = registers[i];
    }
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let num_elems: usize = input.shape.iter().product();

    let handle = client.empty(num_elems * dtype.size());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle, dtype);

    into_contiguous_ref(client, input, &output.as_ref(), dtype)?;

    Ok(output)
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
pub fn into_contiguous_pitched<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec(), dtype);

    into_contiguous_ref(client, input, &output.as_ref(), dtype)?;

    Ok(output)
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
/// Handles unpacking and repacking packed tensors (i.e. quantized values).
/// `shape` refers to the actual (unpacked) shape of the tensor, while `packing` specifies the
/// number of elements in each storage element.
///
/// # Warning
/// This assumes `u32` or `u8` packing.
pub fn into_contiguous_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let rank = shape.len();
    if rank <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let mut out_shape = shape.to_vec();
    out_shape[rank - 1] = out_shape[rank - 1].div_ceil(packing);
    let output = TensorHandle::empty(client, out_shape, dtype);

    // Should reinterpret as u8 if possible at some point, but requires modifying shape/strides so
    // keep it simple for now
    into_contiguous_packed_ref(client, input, &output.as_ref(), shape, packing, dtype)?;

    Ok(output)
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let line_size = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        rank - 1,
    );
    let num_vecs = num_elems / line_size as usize;
    let num_sm = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(NUM_SM_APPROX);
    let cube_dim = CubeDim::new(client, num_vecs);
    let simul_vecs = num_sm * cube_dim.num_elems();
    let mut elems_per_unit = match num_vecs / simul_vecs as usize {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut num_elems_per_unit = line_size as usize * elems_per_unit;

    let last_dim = output.shape[rank - 1];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while !last_dim.is_multiple_of(num_elems_per_unit as usize) {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_vec = if line_size > 1 {
        line_size
    } else {
        *R::supported_line_sizes()
            .iter()
            .filter(|it| num_elems_per_unit.is_multiple_of(**it))
            .max()
            .unwrap_or(&1)
    };

    let input = linear_view(client, input, line_size);
    let out_layout = LinearLayoutArgs::from_handle(client, output, out_vec);

    let cube_count = calculate_cube_count_elemwise(
        client,
        num_elems.div_ceil(num_elems_per_unit as usize),
        cube_dim,
    );

    let launch = if line_size != out_vec && out_vec > 1 {
        into_contiguous_kernel_pack::launch
    } else {
        into_contiguous_kernel::launch
    };

    launch(
        client,
        cube_count,
        cube_dim,
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
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtype),
        output.shape,
        output.strides,
        rank - 1,
    );
    let num_vecs = num_elems / line_size as usize;
    let num_sm = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(NUM_SM_APPROX);
    let cube_dim = CubeDim::new(client, num_vecs);
    let simul_vecs = num_sm * cube_dim.num_elems();
    let mut elems_per_unit = match num_vecs / simul_vecs as usize {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut num_elems_per_unit = line_size as usize * elems_per_unit;

    let last_dim = output.shape[rank - 1];
    let packed_dim = input
        .strides
        .iter()
        .enumerate()
        .rev()
        .find(|(_, s)| **s == 1)
        .expect("At least one stride should be 1")
        .0;

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while !last_dim.is_multiple_of(num_elems_per_unit as usize) {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_layout = LinearLayoutArgs::from_handle(client, output, line_size);

    let cube_count = calculate_cube_count_elemwise(
        client,
        num_elems.div_ceil(num_elems_per_unit as usize),
        cube_dim,
    );

    let in_shape = shape
        .iter()
        .map(|s| FastDivmodArgs::<usize>::new(client, *s))
        .collect();

    into_contiguous_kernel_packed::launch(
        client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        output.as_tensor_arg(line_size),
        out_layout,
        in_shape,
        packed_dim,
        packing,
        rank,
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
