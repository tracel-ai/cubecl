use crate::{
    FastDivmod,
    tensor::{
        TensorHandle, into_contiguous,
        layout::{
            Layout, LayoutExpand,
            linear::{LinearLayout, LinearView, linear_layout, linear_view},
        },
    },
};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise,
    ir::{StorageType, VectorSize},
    tensor_vector_size_parallel,
    zspace::{Strides, strides},
};

pub const NUM_SM_APPROX: u32 = 50;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<T: Scalar, N1: Size, L: Scalar, N2: Size>(
    tensor: &Tensor<Vector<T, N1>>,
    layout: &Tensor<Vector<L, N2>>,
    offset_layout: usize,
    dim_start: usize,
    dim_end: usize,
    #[comptime] unroll: bool,
) -> usize {
    let offset_ref = offset_layout * tensor.vector_size();
    let mut offset = 0;

    #[unroll(unroll)]
    for i in dim_start..dim_end {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / tensor.vector_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous<T: Scalar, N: Size>(
    tensor: &Tensor<Vector<T, N>>,
    offset_layout: usize,
    #[comptime] rank: Option<usize>,
) -> usize {
    let unroll = rank.is_some();
    let rank = rank.unwrap_or_else(|| tensor.rank());

    let offset_ref = offset_layout * tensor.vector_size();
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

    offset / tensor.vector_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous_fastdivmod(
    offset: usize,
    shape: &Sequence<FastDivmod<usize>>,
    stride: &Sequence<usize>,
    #[comptime] vector_size: VectorSize,
) -> usize {
    let rank = shape.len().comptime();

    let offset_ref = offset * vector_size;
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;

        let (rem, ogwl) = shape[dim].div_mod(remainder);
        offset += ogwl * stride[dim];
        remainder = rem;
    }

    offset / vector_size
}

#[cube(launch, address_type = "dynamic")]
fn copy_kernel<T: Numeric, N: Size>(
    input: &LinearView<Vector<T, N>>,
    output: &mut [Vector<T, N>],
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(T)] _elem: StorageType,
) {
    let offset_linear = ABSOLUTE_POS * elems_per_thread;

    let mut registers = Array::<Vector<T, N>>::new(elems_per_thread);

    #[unroll]
    for i in 0..elems_per_thread {
        registers[i] = input.read_checked(offset_linear + i);
    }

    let offset_output = out_layout.to_source_pos(offset_linear);

    #[unroll]
    for i in 0..elems_per_thread {
        write_checked(output, offset_output + i, registers[i]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn copy_kernel_pack<T: Numeric, N: Size>(
    input: &LinearView<T>,
    output: &mut [Vector<T, N>],
    out_layout: LinearLayout,
    #[comptime] elems_per_thread: usize,
    #[define(T)] _elem: StorageType,
) {
    let vector_size = output.vector_size().comptime();
    let vectors_per_thread = elems_per_thread / vector_size;

    let offset_output = ABSOLUTE_POS * vectors_per_thread;
    let offset_input = offset_output * vector_size;

    let mut registers = Array::<Vector<T, N>>::new(vectors_per_thread);

    #[unroll]
    for i in 0..vectors_per_thread {
        let offset = i * vector_size;
        let mut reg = Vector::<T, N>::empty();
        #[unroll]
        for k in 0..vector_size {
            let offset_input = offset_input + offset + k;
            reg.insert(k, input.read_checked(offset_input));
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..vectors_per_thread {
        write_checked(output, offset_output + i, registers[i]);
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
            let dim = rank - i - 1;
            let (rem, mut local_pos) = in_shape[dim].div_mod(remainder);
            remainder = rem;
            if dim == packed_dim {
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

#[cube(launch, address_type = "dynamic")]
fn copy_kernel_packed<T: Int, N: Size>(
    input: &Tensor<T>,
    output: &mut Tensor<Vector<T, N>>,
    out_layout: LinearLayout,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] packed_dim: usize,
    #[comptime] packing: usize,
    #[comptime] rank: usize,
    #[comptime] elems_per_thread: usize,
    #[define(T)] _elem: StorageType,
) {
    let vector_size = output.vector_size().comptime();
    let vectors_per_thread = elems_per_thread / vector_size;

    let offset_output = ABSOLUTE_POS * vectors_per_thread;
    let offset_input = offset_output * vector_size;

    if offset_output >= output.len() {
        terminate!()
    }

    let mut registers = Array::<Vector<T, N>>::new(vectors_per_thread);

    #[unroll]
    for i in 0..vectors_per_thread {
        let offset = i * vector_size;
        let mut reg = Vector::<T, N>::empty();
        #[unroll]
        for k in 0..vector_size {
            let offset_input = offset_input + offset + k;

            reg.insert(
                k,
                index_packed(input, offset_input, &in_shape, packed_dim, packing, rank),
            );
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_source_pos(offset_output);

    #[unroll]
    for i in 0..vectors_per_thread {
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
    input: TensorBinding<R>,
    packed_dim: usize,
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    let rank = shape.len();
    if rank <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let mut out_shape = shape.to_vec();
    out_shape[rank - 1] = out_shape[rank - 1].div_ceil(packing);
    let output = TensorHandle::empty(client, out_shape, dtype);

    // Should reinterpret as u8 if possible at some point, but requires modifying shape/strides so
    // keep it simple for now
    into_contiguous_packed_ref(
        client,
        input,
        output.clone().binding(),
        packed_dim,
        shape,
        packing,
        dtype,
    );

    output
}

/// Make a jit tensor contiguous.
pub fn copy_gpu_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let in_rank = input.strides.len();
    let out_rank = output.strides.len();
    let vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        in_rank - 1,
    );
    let vector_size_out = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        out_rank - 1,
    );
    let vector_size = vector_size_in.min(vector_size_out);

    let num_vecs = num_elems / vector_size as usize;
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

    let mut num_elems_per_unit = vector_size as usize * elems_per_unit;

    let last_dim = output.shape[out_rank - 1];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while !last_dim.is_multiple_of(num_elems_per_unit as usize) {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_vec = if vector_size > 1 {
        vector_size
    } else {
        // Recompute because it needs to account for `num_elems_per_unit`
        client
            .io_optimized_vector_sizes(dtype.size())
            .filter(|it| num_elems_per_unit.is_multiple_of(*it))
            .max()
            .unwrap_or(1)
    };

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));
    let input = linear_view(input);
    let out_layout = linear_layout(&output, out_vec);

    let cube_count = calculate_cube_count_elemwise(
        client,
        num_elems.div_ceil(num_elems_per_unit as usize),
        cube_dim,
    );

    let launch = if vector_size != out_vec && out_vec > 1 {
        copy_kernel_pack::launch
    } else {
        copy_kernel::launch
    };

    launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        out_vec,
        input,
        output.clone().into_buffer_arg(),
        out_layout,
        elems_per_unit,
        dtype,
    )
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_packed_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    packed_dim: usize,
    shape: &[usize],
    packing: usize,
    dtype: StorageType,
) {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let in_rank = input.strides.len();
    let out_rank = output.strides.len();
    let in_packed_dim = in_rank - packed_dim - 1;
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        out_rank - 1,
    );
    let num_vecs = num_elems / vector_size as usize;
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

    let mut num_elems_per_unit = vector_size as usize * elems_per_unit;

    let last_dim = output.shape[out_rank - 1];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while !last_dim.is_multiple_of(num_elems_per_unit as usize) {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_layout = linear_layout(&output, vector_size);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));
    let cube_count = calculate_cube_count_elemwise(
        client,
        num_elems.div_ceil(num_elems_per_unit as usize),
        cube_dim,
    );

    let in_shape = shape.iter().copied().collect();

    copy_kernel_packed::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        output.into_tensor_arg(),
        out_layout,
        in_shape,
        in_packed_dim,
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

    for (&expected, &stride) in compact_strides(shape).iter().zip(strides) {
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

pub fn compact_strides(shape: &[usize]) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
