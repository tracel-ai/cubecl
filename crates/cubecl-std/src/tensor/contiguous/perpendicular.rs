use crate::tensor::{TensorHandle, into_contiguous};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise, ir::VectorSize, tensor_vector_size_parallel,
    tensor_vector_size_perpendicular,
};
use std::cmp::min;

/// Kernel for converting a non-contiguous tensor into a contiguous one when
/// the vectorization axis is perpendicular to the last dimension.
///
/// This kernel handles the case where memory is laid out such that the unit-stride
/// is not on the last dimension, requiring a "gather-and-transpose" pattern
/// to write out contiguous vectors.
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_perpendicular<T: Numeric, N: Size>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    axis_vectorized: usize,
    #[define(T)] _elem: ElemType,
) {
    let vector_size = input.vector_size();
    let last_axis = input.rank() - 1;

    // Calculate how many vectorized vectors fit into the last dimension's shape.
    let num_batch = output.shape(last_axis) / vector_size;

    // Local registers to perform a small in-register transpose.
    let mut accumulators = Sequence::<Vector<T, N>>::new();

    #[unroll]
    for _ in 0..vector_size {
        accumulators.push(Vector::empty());
    }

    let channel_input_stride_elem = input.stride(last_axis);
    let channel_output_stride_elem = output.stride(axis_vectorized);

    // Strides adjusted for vectorization (vector_size).
    let channel_input_stride = channel_input_stride_elem / vector_size;
    let channel_output_stride = channel_output_stride_elem / vector_size;

    // Total parallel units needed to cover the output space.
    let num_runs = output.len() / (num_batch * vector_size);

    if ABSOLUTE_POS >= num_runs {
        terminate!()
    }

    // Mapping the global worker ID to the specific tensor coordinates.
    let batch_index = ABSOLUTE_POS * num_batch;
    let skip_interval = batch_index / channel_output_stride;
    let skip_index = batch_index % channel_output_stride;
    let skip_size = channel_output_stride_elem;
    let global_index = (skip_interval * skip_size) + skip_index;

    for b in 0..num_batch {
        let offset_output = global_index + b;

        // Calculate the physical offset in the input tensor for the current output coordinate.
        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(offset_output * vector_size, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        let batch_offset = batch_offset / vector_size;

        // --- STEP 1: GATHER ---
        // Load data from the input tensor. Since the data is "perpendicular",
        // we read across the stride-1 axis to fill the accumulators.
        #[unroll]
        for i in 0..vector_size {
            let index = batch_offset + i * channel_input_stride;
            let batched = input[index];

            // --- STEP 2: TRANSPOSE ---
            // Rearrange the loaded vector components into the accumulators.
            #[unroll]
            for o in 0..vector_size {
                let vector = accumulators.index_mut(o);
                vector.insert(i, batched.extract(o));
            }
        }

        // --- STEP 3: STORE ---
        // Write the transposed vectors to the output in a contiguous fashion.
        #[unroll]
        for o in 0..vector_size {
            let index_out = offset_output + o * channel_output_stride;
            let batched = accumulators[o];

            output[index_out] = batched;
        }
    }
}

/// The GPU form of [`copy_perpendicular`]: one unit per `vector_size × vector_size` block
/// instead of one per row.
///
/// Each unit gathers `vector_size` vectors along the input's unit-stride axis, transposes them
/// in registers, and stores `vector_size` vectors along the output's last axis. Units walk the
/// output's last axis fastest, so stores are coalesced across a cube, while the loads of
/// neighbouring units along `axis` fill the same cache lines.
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_perpendicular_blocks<T: Numeric, N: Size>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    tiles_last: usize,
    tiles_axis: usize,
    num_tiles: usize,
    #[comptime] axis: usize,
    #[comptime] rank: usize,
    #[define(T)] _elem: ElemType,
) {
    let tile = CUBE_POS;
    if tile >= num_tiles {
        terminate!()
    }

    let vector_size = input.vector_size();
    let last_axis = rank - 1;

    // Tiles are walked diagonally: with power-of-two strides, cubes running side by side on
    // the same input rows would otherwise all hit the same memory channels.
    let tile_in_batch = tile % (tiles_last * tiles_axis);
    let tile_axis = tile_in_batch % tiles_axis;
    let tile_last = (tile_in_batch / tiles_axis + tile_axis) % tiles_last;
    let block_last = tile_last * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    let block_axis = tile_axis * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    if block_last * vector_size >= output.shape(last_axis)
        || block_axis * vector_size >= output.shape(axis)
    {
        terminate!()
    }

    // Every axis but the two being transposed is a batch.
    let mut batch = tile / (tiles_last * tiles_axis);
    let mut offset_input = 0;
    let mut offset_output = 0;
    #[unroll]
    for i in 0..rank {
        let dim = rank - 1 - i;
        if dim != axis && dim != last_axis {
            let coordinate = batch % output.shape(dim);
            batch /= output.shape(dim);
            offset_input += coordinate * input.stride(dim);
            offset_output += coordinate * output.stride(dim);
        }
    }

    let start_axis = block_axis * vector_size;
    let start_last = block_last * vector_size;
    let input_stride = input.stride(last_axis);
    let output_stride = output.stride(axis);

    let index_input = (offset_input + start_axis + start_last * input_stride) / vector_size;
    let index_output = (offset_output + start_axis * output_stride + start_last) / vector_size;
    let input_stride = input_stride / vector_size;
    let output_stride = output_stride / vector_size;

    let mut accumulators = Sequence::<Vector<T, N>>::new();

    #[unroll]
    for _ in 0..vector_size {
        accumulators.push(Vector::empty());
    }

    #[unroll]
    for i in 0..vector_size {
        let gathered = input[index_input + i * input_stride];

        #[unroll]
        for o in 0..vector_size {
            let vector = accumulators.index_mut(o);
            vector.insert(i, gathered.extract(o));
        }
    }

    #[unroll]
    for o in 0..vector_size {
        output[index_output + o * output_stride] = accumulators[o];
    }
}

/// Launches the perpendicular contiguous kernel.
///
/// This is used when the input tensor's memory layout is such that the last dimension
/// is not the one with a stride of 1 (the vectorized dimension). It optimizes
/// the copy by using hardware vectorization (Vectors) and an in-register transpose.
pub fn launch_into_contiguous_perpendicular(
    client: &Client,
    input: TensorBinding,
    dtype: ElemType,
) -> TensorHandle {
    // Fallback for 1D tensors where perpendicularity doesn't apply.
    if input.shape.len() <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec(), dtype);
    launch_copy_perpendicular_ref(client, input, output.clone().binding(), dtype);

    output
}

/// Launches the perpendicular contiguous kernel.
///
/// This is used when the input tensor's memory layout is such that the last dimension
/// is not the one with a stride of 1 (the vectorized dimension). It optimizes
/// the copy by using hardware vectorization (Vectors) and an in-register transpose.
pub fn launch_copy_perpendicular_ref(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
) {
    let (axis, vector_size) = perpendicular_vector_size(client, &input, &output, dtype);
    let rank = output.shape.len();

    let num_elems = output.shape.iter().product::<usize>();
    let working_units = num_elems / (vector_size * output.shape[rank - 1]);
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        copy_perpendicular::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            dtype,
        );
    }
}

/// Launches the GPU perpendicular kernel when its blocks can be vectorized, returning whether
/// it did.
///
/// Without vectorization a block is a single element and the kernel is a strided copy, so the
/// caller keeps its general path for that case.
pub fn try_launch_copy_perpendicular_blocks(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    dtype: ElemType,
) -> bool {
    let (axis, vector_size) = perpendicular_vector_size(client, input, output, dtype);
    if vector_size == 1 {
        return false;
    }

    let rank = output.shape.len();
    let blocks_last = output.shape[rank - 1] / vector_size;
    let blocks_axis = output.shape[axis] / vector_size;
    let batches =
        output.shape.iter().product::<usize>() / (output.shape[rank - 1] * output.shape[axis]);

    // A cube covers up to 8 blocks along the last axis and fills the rest of 256 units along
    // `axis`, so its loads share cache lines while its stores stay coalesced. Narrow outputs
    // shrink the first side rather than leave units idle.
    let cube_x = blocks_last.next_power_of_two().min(8) as u32;
    let cube_dim = CubeDim::new_2d(cube_x, 256 / cube_x);
    let tiles_last = blocks_last.div_ceil(cube_dim.x as usize);
    let tiles_axis = blocks_axis.div_ceil(cube_dim.y as usize);
    let num_tiles = tiles_last * tiles_axis * batches;
    let cube_count = calculate_cube_count_elemwise(client, num_tiles, CubeDim::new_1d(1));
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        copy_perpendicular_blocks::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            tiles_last,
            tiles_axis,
            num_tiles,
            axis,
            rank,
            dtype,
        );
    }

    true
}

/// The input's unit-stride axis, and the vector size both perpendicular kernels can use.
fn perpendicular_vector_size(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    dtype: ElemType,
) -> (usize, VectorSize) {
    let axis = input
        .strides
        .iter()
        .position(|stride| *stride == 1)
        .unwrap_or(0);
    let rank = output.shape.len();

    let vector_size_perpendicular = tensor_vector_size_perpendicular(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        rank - 1,
    );
    let vector_size_parallel = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        rank - 1,
    );
    // The gather loads `vector_size` consecutive elements along the input's unit-stride axis,
    // so that axis must be vectorizable on its own.
    let vector_size_axis = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        axis,
    );
    let vector_size = min(
        min(vector_size_perpendicular, vector_size_parallel),
        vector_size_axis,
    );

    (axis, vector_size)
}
