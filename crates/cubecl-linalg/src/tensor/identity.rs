use cubecl::frontend::TensorHandleRef;
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, tensor_line_size_parallel};
use cubecl_core as cubecl;

use super::TensorHandle;

#[cube(launch_unchecked)]
fn identity_kernel<C: Numeric>(output: &mut Tensor<Line<C>>, gap: u32) {
    if ABSOLUTE_POS < output.len() {
        let mut line = Line::empty(output.line_size()).fill(C::from_int(0));

        let start_pos = ABSOLUTE_POS * output.line_size();
        let mut offset = 0;
        while offset < output.line_size() {
            let remainder = (start_pos + offset) % gap;
            if remainder % gap == 0 {
                line[offset] = C::from_int(1);
                offset += gap;
            } else {
                offset += gap - remainder;
            }
        }
        output[ABSOLUTE_POS] = line;
    }
}

/// Launch identity matrix kernel.
/// Ensure output is a tensorhandle containing a square matrix with shape 2 x 2.
/// output will contain the identity matrix.
pub fn launch<R: Runtime, C: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: &TensorHandle<R, C>,
) {
    launch_ref::<R, C>(client, &output.as_ref());
}

/// Launch identity matrix kernel by ref.
/// Ensure output is a tensorhandle containing a square matrix with shape 2 x 2.
/// output will contain the identity matrix.
pub fn launch_ref<R: Runtime, C: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: &TensorHandleRef<R>,
) {
    assert_eq!(2, output.shape.len(), "input should be a matrix");
    assert_eq!(
        output.shape[0], output.shape[1],
        "input should be a square matrix"
    );

    let num_elements: usize = output.shape.iter().product();
    let rank = output.shape.len();

    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        output.shape,
        output.strides,
        rank - 1,
    );

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elements / vectorization_factor as usize, cube_dim);

    unsafe {
        identity_kernel::launch_unchecked::<C, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<C>(
                output.handle,
                output.strides,
                output.shape,
                vectorization_factor,
            ),
            ScalarArg::new(output.shape[0] as u32 + 1),
        );
    }
}
