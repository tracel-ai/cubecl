use std::cmp::min;

use crate::tensor::{TensorHandle, into_contiguous};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel,
    tensor_line_size_perpendicular,
};

#[cube(launch_unchecked)]
pub fn into_contiguous_perpendicular<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    axis_vectorized: u32,
    #[define(N)] _elem: StorageType,
) {
    let line_size = input.line_size();

    let last_axis = input.rank() - 1;
    let num_batch = output.shape(last_axis) / line_size;

    let mut accumulators = Sequence::<Line<N>>::new();

    #[unroll]
    for _ in 0..line_size {
        accumulators.push(Line::empty(line_size));
    }

    let channel_input_stride_elem = input.stride(last_axis);
    let channel_output_stride_elem = output.stride(axis_vectorized);

    let channel_input_stride = channel_input_stride_elem / line_size;
    let channel_output_stride = channel_output_stride_elem / line_size;

    // debug_print!("channel_input_stride %i\n", channel_input_stride);
    // debug_print!("channel_output_stride %i\n", channel_output_stride);

    let num_runs = output.len() / (num_batch * line_size);

    if ABSOLUTE_POS >= num_runs {
        terminate!()
    }

    let batch_index = ABSOLUTE_POS * num_batch;
    let skip_interval = batch_index / channel_output_stride;
    let skip_index = batch_index % channel_output_stride;
    let skip_size = channel_output_stride_elem; // `channel_output_stride * line_size`
    let global_index = (skip_interval * skip_size) + skip_index;

    // debug_print!("global_index %i - ", global_index);
    // debug_print!("skip_interval %i - ", skip_interval);
    // debug_print!("skip_size %i - ", skip_size);
    // debug_print!("skip_index %i - ", skip_index);
    // debug_print!("ABSOLUTE_POS %i\n", ABSOLUTE_POS);

    for b in 0..num_batch {
        let offset_output = global_index + b;

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(offset_output * line_size, axis);

            batch_offset += coordinate * input.stride(axis);
        }
        let batch_offset = batch_offset / line_size;

        for i in 0..line_size {
            let index = batch_offset + i * channel_input_stride;
            let batched = input[index];

            #[unroll]
            for o in 0..line_size {
                let line = accumulators.index_mut(o);
                line[i] = batched[o];
            }
        }

        #[unroll]
        for o in 0..line_size {
            let index_out = offset_output + o * channel_output_stride;
            let batched = *accumulators.index(o);

            output[index_out] = batched;
        }
    }
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
pub fn into_contiguous_lined_own<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec(), dtype);

    into_contiguous_lined_ref(client, input, &output.as_ref(), dtype)?;

    Ok(output)
}

pub fn into_contiguous_lined_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let mut axis = 0;

    for (i, stride) in input.strides.iter().enumerate() {
        if *stride == 1 {
            axis = i;
            break;
        }
    }
    let rank = output.shape.len();

    let line_size_perpendicular = tensor_line_size_perpendicular(
        client.io_optimized_line_sizes(&dtype),
        input.shape,
        input.strides,
        rank - 1,
    );
    let line_size_parallel = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtype),
        output.shape,
        output.strides,
        rank - 1,
    );
    let line_size = min(line_size_perpendicular, line_size_parallel);

    let num_elems = output.shape.iter().product::<usize>();
    let working_units = num_elems / (line_size as usize * output.shape[rank - 1]);
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    println!("Axis {axis:?} | {cube_dim:?} | {cube_count:?}");
    println!("{:?} {:?}", input.shape, input.strides);
    println!("{:?} {:?}", output.shape, output.strides);

    unsafe {
        into_contiguous_perpendicular::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
            ScalarArg::new(axis as u32),
            dtype,
        )?;
    }

    Ok(())
}
