use super::TensorHandle;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size};

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

#[cube(launch_unchecked)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    #[comptime] rank: Option<u32>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    let offset_input = index_offset_with_layout::<N, N>(
        input,
        output,
        offset_output,
        0,
        rank.unwrap_or_else(|| output.rank()),
        rank.is_some(),
    );

    output[offset_output] = input[offset_input];
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let vectorization_factor = tensor_line_size(
        R::supported_line_sizes(),
        input.shape,
        input.strides,
        rank - 1,
    );

    let num_elems: usize = input.shape.iter().product();
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);
    let handle = client.empty(num_elems * E::as_elem().size());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle);

    unsafe {
        into_contiguous_kernel::launch_unchecked::<Line<E>, R>(
            client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(vectorization_factor),
            output.as_ref().as_tensor_arg(vectorization_factor),
            Some(rank as u32),
        );
    }

    output
}
