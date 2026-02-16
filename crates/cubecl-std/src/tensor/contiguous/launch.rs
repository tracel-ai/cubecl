use crate::tensor::{TensorHandle, copy_gpu_ref, launch_copy_perpendicular_ref};
use cubecl_core::{
    Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef, server::LaunchError,
};

/// Make a jit tensor contiguous.
pub fn into_contiguous_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let elem_count: usize = input.shape.iter().product();

    let handle = client.empty(elem_count * dtype.size());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle, dtype);

    copy_into(client, input, &output.as_ref(), dtype)?;

    Ok(output)
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [`create_tensor`](cubecl_runtime::client::ComputeClient::create_tensor).
pub fn into_contiguous_pitched_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    if input.shape.len() <= 1 {
        return into_contiguous_ref(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec(), dtype);

    copy_into(client, input, &output.as_ref(), dtype)?;

    Ok(output)
}

/// Copies the input tensor into the output tensor following the strides.
pub fn copy_into<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let rank = input.strides.len();

    // It's normally faster on all devices, but since it doesn't parallelize on an axis, it
    // might be worst on GPU. Should tune at some point.
    let is_cpu = client.properties().hardware.cpu_core_count.is_some();
    if input.strides[rank - 1] != 1 && is_cpu {
        launch_copy_perpendicular_ref(client, input, output, dtype)?;
    } else {
        copy_gpu_ref(client, input, output, dtype)?;
    };

    Ok(())
}
