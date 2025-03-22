use super::TensorHandle;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};

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

#[cube(launch)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &Tensor<Line<N>>,
    layout_ref: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    #[comptime] rank: Option<u32>,
    #[comptime] elems_per_thread: u32,
    #[comptime] pitched: bool,
) {
    let const_rank = rank.is_some();
    let rank = rank.unwrap_or_else(|| output.rank());
    let mut offset_output = ABSOLUTE_POS * elems_per_thread;
    let line_size = input.line_size();

    let mut registers = Array::vectorized(elems_per_thread, line_size);

    #[unroll]
    for i in 0..elems_per_thread {
        let offset_input = index_offset_with_layout::<N, N>(
            input,
            layout_ref,
            offset_output + i,
            0,
            rank,
            const_rank,
        );

        registers[i] = input[offset_input];
    }

    if pitched {
        let offset_abs = offset_output * line_size;
        let x = offset_abs % output.shape(rank - 1);
        let y = offset_abs / output.shape(rank - 1);
        offset_output = y * output.stride(rank - 2) + x;
    }

    #[unroll]
    for i in 0..elems_per_thread {
        output[offset_output + i] = registers[i];
    }
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    let num_elems: usize = input.shape.iter().product();
    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        rank - 1,
    );
    let num_vecs = num_elems / vectorization_factor as usize;
    let approx_sm = 64;
    let approx_simul_vecs = approx_sm * CubeDim::default().num_elems();
    let elems_per_unit = match num_vecs as u32 / approx_simul_vecs {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    // TODO: Benchmark to find good default prefetch, for now preserve existing behaviour
    into_contiguous_prefetch(client, input, elems_per_unit, false)
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_pitched<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input);
    }

    let num_elems: usize = input.shape.iter().product();
    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        rank - 1,
    );
    let num_vecs = num_elems / vectorization_factor as usize;
    let approx_sm = 64;
    let approx_simul_vecs = approx_sm * CubeDim::default().num_elems();
    let elems_per_unit = match num_vecs as u32 / approx_simul_vecs {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    // TODO: Benchmark to find good default prefetch, for now preserve existing behaviour
    into_contiguous_prefetch(client, input, elems_per_unit, true)
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_prefetch<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    mut elems_per_unit: u32,
    pitched: bool,
) -> TensorHandle<R, E> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        rank - 1,
    );

    let num_elems: usize = input.shape.iter().product();
    let output = if pitched {
        TensorHandle::empty(client, input.shape.to_vec())
    } else {
        let handle = client.empty(num_elems * size_of::<E>());
        TensorHandle::new_contiguous(input.shape.to_vec(), size_of::<E>(), handle)
    };

    let mut num_elems_per_unit = vectorization_factor as u32 * elems_per_unit;

    let last_dim = output.shape()[rank - 1];
    let is_padded = rank > 1 && last_dim != output.strides()[rank - 2];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while is_padded && last_dim % num_elems_per_unit as usize != 0 {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(num_elems_per_unit as usize), cube_dim);

    let out_handle = output.handle.handle.clone();
    let layout_strides = compact_strides(output.shape());

    let layout_ref = if is_padded {
        unsafe {
            TensorArg::from_raw_parts::<E>(
                &out_handle,
                &layout_strides,
                output.shape(),
                vectorization_factor,
            )
        }
    } else {
        TensorArg::alias(2)
    };

    into_contiguous_kernel::launch::<Line<E>, R>(
        client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(vectorization_factor),
        layout_ref,
        output.as_ref().as_tensor_arg(vectorization_factor),
        Some(rank as u32),
        elems_per_unit,
        is_padded,
    );

    output
}

pub fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
