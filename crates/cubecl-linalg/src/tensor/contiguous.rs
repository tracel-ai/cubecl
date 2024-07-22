use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise, tensor_vectorization_factor, SUBCUBE_DIM_APPROX,
};

use cubecl::prelude::*;

use super::TensorHandle;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    tensor: &Tensor<N>,
    layout: &Tensor<L>,
    offset_layout: UInt,
    dim_start: UInt,
    dim_end: UInt,
    unroll: Comptime<bool>,
) -> UInt {
    let vectorization_factor = Comptime::vectorization(tensor);
    let vectorization_factor_runtime = Comptime::runtime(vectorization_factor);

    let offset_ref = offset_layout * vectorization_factor_runtime;
    let mut offset = UInt::new(0);

    for i in range(dim_start, dim_end, unroll) {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / vectorization_factor_runtime
}

#[cube(launch)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &Tensor<N>,
    output: &mut Tensor<N>,
    rank: Comptime<Option<UInt>>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    let offset_input = index_offset_with_layout::<N, N>(
        input,
        output,
        offset_output,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
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
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &input.shape, &input.strides, rank - 1);

    let num_elems: usize = input.shape.iter().product();
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );
    let handle = client.empty(num_elems * E::as_elem().size());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle);

    into_contiguous_kernel::launch::<E, R>(
        &client,
        cube_count,
        CubeDim::default(),
        TensorArg::vectorized(
            vectorization_factor,
            &input.handle,
            &input.strides,
            &input.shape,
        ),
        TensorArg::vectorized(
            vectorization_factor,
            &output.handle,
            &output.strides,
            &output.shape,
        ),
        Some(UInt::new(rank as u32)),
    );

    output
}
