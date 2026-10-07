use crate::tensor::{TensorHandle, copy_gpu_ref, launch_copy_perpendicular_ref};
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

    // It's normally faster on all devices, but since it doesn't parallelize on an axis, it
    // might be worst on GPU. Should tune at some point.
    let is_cpu = client.properties().hardware.num_cpu_cores.is_some();
    // The perpendicular kernel reads the input at the output's coordinates, so it only
    // expresses a change of layout. A reshape keeping the rank would otherwise reach it and
    // index the input out of bounds.
    let same_shape = input.shape == output.shape;
    if input.strides[rank - 1] != 1 && is_cpu && same_shape {
        launch_copy_perpendicular_ref(client, input, output, dtype);
    } else {
        copy_gpu_ref(client, input, output, dtype);
    };
}
