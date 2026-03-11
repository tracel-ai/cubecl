use crate::tensor::{TensorHandle, into_contiguous};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise, tensor_vector_size_parallel,
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
    #[define(T)] _elem: StorageType,
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
        for i in 0..vector_size {
            let index = batch_offset + i * channel_input_stride;
            let batched = input[index];

            // --- STEP 2: TRANSPOSE ---
            // Rearrange the loaded vector components into the accumulators.
            #[unroll]
            for o in 0..vector_size {
                let vector = accumulators.index_mut(o);
                vector[i] = batched[o];
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

/// Launches the perpendicular contiguous kernel.
///
/// This is used when the input tensor's memory layout is such that the last dimension
/// is not the one with a stride of 1 (the vectorized dimension). It optimizes
/// the copy by using hardware vectorization (Vectors) and an in-register transpose.
pub fn launch_into_contiguous_perpendicular<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    dtype: StorageType,
) -> TensorHandle<R> {
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
pub fn launch_copy_perpendicular_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) {
    let mut axis = 0;

    for (i, stride) in input.strides.iter().enumerate() {
        if *stride == 1 {
            axis = i;
            break;
        }
    }
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
    let vector_size = min(vector_size_perpendicular, vector_size_parallel);

    let num_elems = output.shape.iter().product::<usize>();
    let working_units = num_elems / (vector_size as usize * output.shape[rank - 1]);
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        copy_perpendicular::launch_unchecked::<R>(
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
