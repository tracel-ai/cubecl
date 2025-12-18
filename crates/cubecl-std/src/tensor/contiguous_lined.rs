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
    let stride = output.stride(axis);
    let line_size = input.line_size();

    let mut accumulators = Sequence::<Line<N>>::new();

    #[unroll]
    for _ in 0..line_size {
        accumulators.push(Line::empty(line_size));
    }

    let index_start = ABSOLUTE_POS * line_size;

    for i in 0..line_size {
        let index_read = index_start + i;
        let batched = input[index_read];

        #[unroll]
        for o in 0..line_size {
            let line = accumulators.index_mut(o);
            line[i] = batched[o];
        }
    }

    let index_start = index_start * stride;

    #[unroll]
    for o in 0..line_size {
        let index_out = index_start + o;

        output[index_out] = *accumulators.index(o);
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
    // Vectorization is only enabled when the last dimension is contiguous.
    let mut line_size = tensor_line_size_perpendicular(
        client.io_optimized_line_sizes(&dtype),
        input.shape,
        input.strides,
        axis,
    );
    let rank = output.shape.len();
    let out_size = output.shape[rank - 1];

    if line_size as usize % out_size != 0 {
        for ls in client.io_optimized_line_sizes(&dtype) {
            if ls < line_size && ls as usize % out_size == 0 {
                line_size = ls;
                break;
            }
        }
    }

    let num_elems = input.shape.iter().product::<usize>();
    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

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
