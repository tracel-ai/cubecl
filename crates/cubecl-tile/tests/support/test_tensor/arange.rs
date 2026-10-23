use cubecl::{
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
    zspace::{Shape, Strides},
};

use crate::support::test_tensor::base::BaseInputSpec;

#[cube(launch)]
fn arange_launch<T: Numeric>(tensor: &mut Tensor<T>, #[define(T)] _types: ElemType) {
    let linear = ABSOLUTE_POS;

    if linear >= tensor.len() {
        terminate!();
    }

    let mut remaining = linear;
    let mut offset = 0;

    for d in 0..tensor.rank() {
        let dim = tensor.shape(tensor.rank() - 1 - d);
        let idx = remaining % dim;
        remaining /= dim;
        offset += idx * tensor.stride(tensor.rank() - 1 - d);
    }

    tensor.write_checked(offset, T::cast_from(linear));
}

fn new_arange(client: &Client, shape: Shape, strides: Strides, dtype: ElemType) -> TensorHandle {
    let num_elems = shape.iter().product::<usize>();

    // Performance is not important here and this simplifies greatly the problem
    let vector_size = 1;

    let working_units: u32 = num_elems as u32 / vector_size as u32;
    let cube_dim = CubeDim::new(client, working_units as usize);
    let cube_count = working_units.div_ceil(cube_dim.num_elems());

    let out = TensorHandle::new(
        client.empty(dtype.size() * num_elems),
        shape,
        strides,
        dtype,
    );

    arange_launch::launch(
        client,
        CubeCount::new_1d(cube_count),
        cube_dim,
        out.clone().into_arg(),
        dtype,
    );

    out
}

pub(crate) fn build_arange(spec: BaseInputSpec) -> TensorHandle {
    new_arange(&spec.client, spec.shape.clone(), spec.strides(), spec.dtype)
}
