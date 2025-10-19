use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{
        Layout, LayoutExpand,
        linear::{LinearLayout, LinearLayoutArgs, LinearView, linear_view},
    },
};

use super::TensorHandle;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};

pub const NUM_SM_APPROX: u32 = 50;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    layout: &Tensor<Line<L>>,
    offset_layout: u32,
    dim_start: u32,
    dim_end: u32,
    #[comptime] unroll: bool,
) -> u32 {
    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;

    #[unroll(unroll)]
    for i in dim_start..dim_end {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous<N: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    offset_layout: u32,
    #[comptime] rank: Option<u32>,
) -> u32 {
    let unroll = rank.is_some();
    let rank = rank.unwrap_or_else(|| tensor.rank());

    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll(unroll)]
    for i in 0..rank {
        let dim = rank - i - 1;
        let shape = tensor.shape(dim);
        let ogwl = remainder % shape;
        offset += ogwl * tensor.stride(dim);
        remainder /= shape;
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous_fastdivmod(
    offset: u32,
    shape: &Sequence<FastDivmod>,
    stride: &Sequence<u32>,
    #[comptime] line_size: u32,
) -> u32 {
    let rank = comptime![shape.len()];

    let offset_ref = offset * line_size;
    let mut offset = 0;
    let mut remainder = offset_ref;

    let mut dim = comptime![rank - 1];

    #[unroll]
    for _ in 0..rank {
        let shape = shape.index(dim);
        let (rem, ogwl) = shape.div_mod(remainder);
        offset += ogwl * stride.index(dim);
        remainder = rem;

        comptime![dim = dim.saturating_sub(1);]
    }

    offset / line_size
}

#[cube(launch)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: u32,
) {
    let offset_output = ABSOLUTE_POS * elems_per_thread;
    let line_size = input.line_size();

    let mut registers = Array::<Line<N>>::vectorized(elems_per_thread, line_size);

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
fn into_contiguous_kernel_pack<N: CubePrimitive>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: u32,
) {
    let line_size = output.line_size();
    let lines_per_thread = comptime![elems_per_thread / line_size];

    let offset_output = ABSOLUTE_POS * lines_per_thread;
    let offset_input = offset_output * line_size;

    let mut registers = Array::<Line<N>>::vectorized(lines_per_thread, line_size);

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

#[cube]
fn index_packed<N: Int>(
    tensor: &Tensor<N>,
    pos: u32,
    in_shape: &Sequence<FastDivmod>,
    #[comptime] packed_dim: u32,
    #[comptime] packing: u32,
    #[comptime] rank: u32,
) -> N {
    let bits_per_elem = comptime![N::elem_size_bits() / packing];
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
    in_shape: Sequence<FastDivmod>,
    #[comptime] packed_dim: u32,
    #[comptime] packing: u32,
    #[comptime] rank: u32,
    #[comptime] elems_per_thread: u32,
) {
    let line_size = output.line_size();
    let lines_per_thread = comptime![elems_per_thread / line_size];

    let offset_output = ABSOLUTE_POS * lines_per_thread;
    let offset_input = offset_output * line_size;

    if offset_output >= output.len() {
        terminate!()
    }

    let mut registers = Array::<Line<N>>::vectorized(lines_per_thread, line_size);

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
pub fn into_contiguous<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    let num_elems: usize = input.shape.iter().product();

    let handle = client.empty(num_elems * size_of::<E>());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle);

    into_contiguous_ref::<R, E>(client, input, &output.as_ref());

    output
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
pub fn into_contiguous_pitched<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec());

    into_contiguous_ref::<R, E>(client, input, &output.as_ref());

    output
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
/// Handles unpacking and repacking packed tensors (i.e. quantized values).
/// `shape` refers to the actual (unpacked) shape of the tensor, while `packing` specifies the
/// number of elements in each storage element.
///
/// # Warning
/// This assumes `u32` or `u8` packing.
pub fn into_contiguous_packed<R: Runtime, I: Int>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandleRef<'_, R>,
    shape: &[usize],
    packing: u32,
) -> TensorHandle<R, I> {
    let rank = shape.len();
    if rank <= 1 {
        return into_contiguous(client, input);
    }

    let mut out_shape = shape.to_vec();
    out_shape[rank - 1] = out_shape[rank - 1].div_ceil(packing as usize);
    let output = TensorHandle::<R, I>::empty(client, out_shape);

    // Should reinterpret as u8 if possible at some point, but requires modifying shape/strides so
    // keep it simple for now
    into_contiguous_packed_ref::<R, I>(client, input, &output.as_ref(), shape, packing);

    output
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_ref<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
) {
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
    let simul_vecs = num_sm * CubeDim::default().num_elems();
    let mut elems_per_unit = match num_vecs as u32 / simul_vecs {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut num_elems_per_unit = line_size as u32 * elems_per_unit;

    let last_dim = output.shape[rank - 1];
    let is_padded = rank > 1 && last_dim != output.strides[rank - 2];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while is_padded && !last_dim.is_multiple_of(num_elems_per_unit as usize) {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_vec = if line_size > 1 {
        line_size
    } else {
        *R::supported_line_sizes()
            .iter()
            .filter(|it| num_elems_per_unit.is_multiple_of(**it as u32))
            .max()
            .unwrap_or(&1)
    };

    let input = linear_view(client, input, line_size);
    let out_layout = LinearLayoutArgs::from_handle(client, output, out_vec);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(num_elems_per_unit as usize), cube_dim);

    let launch = if line_size != out_vec && out_vec > 1 {
        into_contiguous_kernel_pack::launch::<E, R>
    } else {
        into_contiguous_kernel::launch::<E, R>
    };

    launch(
        client,
        cube_count,
        cube_dim,
        input,
        output.as_tensor_arg(out_vec),
        out_layout,
        elems_per_unit,
    );
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_packed_ref<R: Runtime, E: Int>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    shape: &[usize],
    packing: u32,
) {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let line_size = tensor_line_size_parallel(
        R::io_optimized_line_sizes(&E::as_type_native_unchecked()),
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
    let simul_vecs = num_sm * CubeDim::default().num_elems();
    let mut elems_per_unit = match num_vecs as u32 / simul_vecs {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut num_elems_per_unit = line_size as u32 * elems_per_unit;

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

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(num_elems_per_unit as usize), cube_dim);

    let in_shape = shape
        .iter()
        .map(|s| FastDivmodArgs::new(client, *s as u32))
        .collect();

    into_contiguous_kernel_packed::launch::<E, R>(
        client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        output.as_tensor_arg(line_size),
        out_layout,
        in_shape,
        packed_dim as u32,
        packing,
        rank as u32,
        elems_per_unit,
    );
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
