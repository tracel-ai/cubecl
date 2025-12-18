use crate::tensor::{TensorHandle, into_contiguous};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_perpendicular};

#[cube(launch_unchecked)]
pub fn into_contiguous_lined<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    axis: u32,
    #[define(N)] _elem: StorageType,
) {
    let line_size = input.line_size();
    let num_writes = line_size;
    let offset_output = ABSOLUTE_POS;

    // if offset_output + line_size >= output.len() {
    //     terminate!()
    // }

    let mut accumulators = Sequence::<Line<N>>::new();

    #[unroll]
    for _ in 0..line_size {
        accumulators.push(Line::empty(line_size));
    }

    let mut batch_offset = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(offset_output * num_writes, axis);
        batch_offset += coordinate * input.stride(axis);
    }
    batch_offset = batch_offset.div_ceil(line_size);

    debug_print!("Absolute pos %i\n", offset_output);
    debug_print!("Batch offset %i\n", batch_offset);
    let vector_stride = input.stride(input.rank() - 1) / line_size;

    for i in 0..line_size {
        debug_print!("Line %i ", i);
        let index = batch_offset + i * vector_stride;
        debug_print!(" - Index %i:", index);
        let batched = input[index];
        let item1 = u32::cast_from(batched[0u32]);
        let item2 = u32::cast_from(batched[1u32]);
        debug_print!(" [%i, ", item1);
        debug_print!(" %i]\n", item2);

        #[unroll]
        for o in 0..line_size {
            let line = accumulators.index_mut(o);
            line[i] = batched[o];
        }
    }

    let vector_stride = output.stride(axis) / line_size;
    debug_print!("Output strides %i \n", vector_stride);

    #[unroll]
    for o in 0..line_size {
        let index_out = offset_output + o * vector_stride;
        debug_print!("Output index %i", index_out);
        let batched = *accumulators.index(o);

        let item1 = u32::cast_from(batched[0u32]);
        let item2 = u32::cast_from(batched[1u32]);
        debug_print!(" [%i, ", item1);
        debug_print!(" %i]\n", item2);

        output[index_out] = batched;
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
    let out_size = output.shape[rank - 1];

    let mut line_size = tensor_line_size_perpendicular(
        client.io_optimized_line_sizes(&dtype),
        input.shape,
        input.strides,
        rank - 1,
    );
    line_size = 2u8;
    println!("{line_size:?} - {out_size:?}");

    if (line_size as usize).pow(2) % out_size != 0 {
        for ls in client.io_optimized_line_sizes(&dtype) {
            if ls < line_size && (ls as usize).pow(2) % out_size == 0 {
                line_size = ls;
                break;
            }
        }
    }

    let num_elems = input.shape.iter().product::<usize>();
    let working_units = num_elems / (line_size as usize * line_size as usize);
    let cube_dim = CubeDim::new(client, working_units);
    let cube_dim = CubeDim::new_single();
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    println!("Axis {axis:?}{cube_dim:?} {cube_count:?} {line_size:?}");
    println!("{:?} {:?}", input.shape, input.strides);
    println!("{:?} {:?}", output.shape, output.strides);

    unsafe {
        into_contiguous_lined::launch_unchecked::<R>(
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
