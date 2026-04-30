use cubecl::frontend::TensorBinding;
use cubecl::prelude::*;
use cubecl::tensor_vector_size_parallel;
use cubecl_core as cubecl;

use super::TensorHandle;

#[cube(launch_unchecked, address_type = "dynamic")]
fn identity_kernel<C: Numeric, N: Size>(
    output: &mut Tensor<Vector<C, N>>,
    gap: usize,
    #[define(C)] _elem: StorageType,
) {
    let pos_x = ABSOLUTE_POS_X as usize * output.vector_size();
    let pos_y = ABSOLUTE_POS_Y as usize;
    let vector_size = output.vector_size();
    if pos_y < output.shape(0) && pos_x < output.shape(1) {
        let mut vector = Vector::new(C::from_int(0));
        let offs_y = pos_y * output.stride(0);

        let start_pos = offs_y + pos_x;
        let mut offset = 0;
        while offset < output.vector_size() {
            let remainder = (start_pos + offset) % gap;
            if remainder == 0 {
                vector.insert(offset, C::from_int(1));
                offset += gap;
            } else {
                offset += gap - remainder;
            }
        }
        output[start_pos / vector_size] = vector;
    }
}

/// Launch identity matrix kernel.
/// Ensure output is a [`TensorHandle`] containing a square matrix.
/// output will contain the identity matrix.
pub fn launch<R: Runtime>(client: &ComputeClient<R>, output: &TensorHandle<R>) {
    let dtype = output.dtype;
    launch_ref(client, output.clone().binding(), dtype);
}

/// Launch identity matrix kernel by ref.
/// Ensure output is a [`TensorHandleRef`] containing a square matrix.
/// output will contain the identity matrix.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) {
    assert_eq!(2, output.shape.len(), "input should be a matrix");
    assert_eq!(
        output.shape[0], output.shape[1],
        "input should be a square matrix"
    );

    let vectorization_factor = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        1,
    );

    let cube_dim = CubeDim::new_2d(16, 16);
    let vectors_x = output.shape[1] as u32 / vectorization_factor as u32;
    let cube_count_x = vectors_x.div_ceil(cube_dim.x);
    let cube_count_y = (output.shape[0] as u32).div_ceil(cube_dim.y);
    let cube_count = CubeCount::new_2d(cube_count_x, cube_count_y);

    let scalar = output.strides[0] + 1;
    unsafe {
        identity_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            output.required_address_type(dtype.size()),
            vectorization_factor,
            output.into_tensor_arg(),
            scalar,
            dtype,
        )
    }
}
