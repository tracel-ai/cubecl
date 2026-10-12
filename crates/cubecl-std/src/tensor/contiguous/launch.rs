use crate::tensor::{
    TensorHandle, copy_gpu_ref, launch_copy_perpendicular_ref, try_launch_copy_perpendicular_blocks,
};
use cubecl_core::{client::Client, ir::ElemType, prelude::TensorBinding};

/// Make a jit tensor contiguous.
pub fn into_contiguous(client: &Client, input: TensorBinding, dtype: ElemType) -> TensorHandle {
    let num_elems: usize = input.shape.iter().product();

    let handle = client.empty(num_elems * dtype.size());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle, dtype);

    copy_into(client, input, output.clone().binding(), dtype);

    output
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [`create_tensor`](cubecl_runtime::client::Client::create_tensor).
pub fn into_contiguous_pitched(
    client: &Client,
    input: TensorBinding,
    dtype: ElemType,
) -> TensorHandle {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.clone(), dtype);

    copy_into(client, input, output.clone().binding(), dtype);

    output
}

/// Copies the input tensor into the output tensor following the strides.
pub fn copy_into(client: &Client, input: TensorBinding, output: TensorBinding, dtype: ElemType) {
    let rank = input.strides.len();

    // The perpendicular kernel reads the input at the output's coordinates, so it only
    // expresses a change of layout. A reshape keeping the rank would otherwise reach it and
    // index the input out of bounds.
    let same_shape = input.shape == output.shape;
    if input.strides[rank - 1] == 1 || !same_shape {
        return copy_gpu_ref(client, input, output, dtype);
    }

    // A CPU unit walks a whole row, which suits it. A GPU needs one unit per block, and only
    // gains over the general copy when the block is vectorized.
    if client.properties().hardware.num_cpu_cores.is_some() {
        launch_copy_perpendicular_ref(client, input, output, dtype);
    } else if !try_launch_copy_perpendicular_blocks(client, &input, &output, dtype) {
        copy_gpu_ref(client, input, output, dtype);
    }
}
