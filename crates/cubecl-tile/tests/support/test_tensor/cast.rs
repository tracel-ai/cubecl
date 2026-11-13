use cubecl::{
    prelude::*,
    std::tensor::{
        TensorHandle, ViewOperations, ViewOperationsExpand, ViewOperationsMut,
        ViewOperationsMutExpand,
    },
    tensor_vector_size_parallel,
    zspace::{shape, strides},
};

#[cube(launch)]
fn cast_launch<From: Numeric, To: Numeric, N: Size>(
    from: &Tensor<Vector<From, N>>,
    to: &mut Tensor<Vector<To, N>>,
    #[define(From, To)] _types: [ElemType; 2],
) {
    cast_inner::<From, To, N>(from, to);
}

#[cube]
fn cast_inner<From: Numeric, To: Numeric, N: Size>(
    from: &Tensor<Vector<From, N>>,
    to: &mut Tensor<Vector<To, N>>,
) {
    to.write_checked(
        ABSOLUTE_POS,
        Vector::cast_from(from.read_checked(ABSOLUTE_POS)),
    )
}

pub fn copy_casted(client: &Client, original: TensorHandle, target_type: ElemType) -> TensorHandle {
    if target_type == original.dtype {
        return TensorHandle::new_contiguous(
            original.shape().clone(),
            original.handle.clone(),
            target_type,
        );
    }

    let num_elems: usize = original.shape().num_elements();

    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(target_type.size()),
        &shape![num_elems],
        &strides![1],
        0,
    );

    let working_units: u32 = num_elems as u32 / vector_size as u32;
    let cube_dim = CubeDim::new(client, working_units as usize);
    let cube_count = working_units.div_ceil(cube_dim.num_elems());

    let out = TensorHandle::new_contiguous(
        original.shape().clone(),
        client.empty(target_type.size() * num_elems),
        target_type,
    );

    let dtype = original.dtype;

    cast_launch::launch(
        client,
        CubeCount::Static(cube_count, 1, 1),
        cube_dim,
        vector_size,
        original.into_arg(),
        out.clone().into_arg(),
        [dtype, target_type],
    );

    out
}
