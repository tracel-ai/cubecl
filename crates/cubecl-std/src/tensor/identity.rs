use cubecl::frontend::TensorHandleRef;
use cubecl::prelude::*;
use cubecl::tensor_line_size_parallel;
use cubecl_core as cubecl;

use super::TensorHandle;

#[cube(launch_unchecked)]
fn identity_kernel<C: Numeric>(
    output: &mut Tensor<Line<C>>,
    gap: u32,
    #[define(C)] _elem: StorageType,
) {
    let pos_x = ABSOLUTE_POS_X * output.line_size();
    if ABSOLUTE_POS_Y < output.shape(0) && pos_x < output.shape(1) {
        let mut line = Line::empty(output.line_size()).fill(C::from_int(0));
        let offs_y = ABSOLUTE_POS_Y * output.stride(0);

        let start_pos = offs_y + pos_x;
        let mut offset = 0;
        while offset < output.line_size() {
            let remainder = (start_pos + offset) % gap;
            if remainder == 0 {
                line[offset] = C::from_int(1);
                offset += gap;
            } else {
                offset += gap - remainder;
            }
        }
        output[start_pos / output.line_size()] = line;
    }
}

/// Launch identity matrix kernel.
/// Ensure output is a [`TensorHandle`] containing a square matrix.
/// output will contain the identity matrix.
pub fn launch<R: Runtime>(client: &ComputeClient<R>, output: &TensorHandle<R>) {
    let dtype = output.dtype;
    launch_ref(client, &output.as_ref(), dtype);
}

/// Launch identity matrix kernel by ref.
/// Ensure output is a [`TensorHandleRef`] containing a square matrix.
/// output will contain the identity matrix.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    output: &TensorHandleRef<R>,
    dtype: StorageType,
) {
    assert_eq!(2, output.shape.len(), "input should be a matrix");
    assert_eq!(
        output.shape[0], output.shape[1],
        "input should be a square matrix"
    );

    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        output.shape,
        output.strides,
        1,
    );

    let cube_dim = CubeDim::new_1d(1);
    let lines_x = output.shape[1] as u32 / vectorization_factor as u32;
    let cube_count_x = lines_x.div_ceil(cube_dim.x);
    let cube_count_y = (output.shape[0] as u32).div_ceil(cube_dim.y);
    let cube_count = CubeCount::new_2d(cube_count_x, cube_count_y);

    unsafe {
        identity_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts_and_size(
                output.handle,
                output.strides,
                output.shape,
                vectorization_factor,
                dtype.size(),
            ),
            ScalarArg::new(output.strides[0] as u32 + 1),
            dtype,
        )
        .expect("Should be able to launch the kernel all the time")
    }
}
