use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise};
use cubecl_linalg::tensor::{into_contiguous, TensorHandle};

use crate::ConvOptions;

#[derive(CubeLaunch)]
struct Im2ColArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: u32,
    padding_w: u32,

    kernel_h: u32,
    kernel_w: u32,
    out_h: u32,
    out_w: u32,

    col_size_1: u32,
    num_elements: u32,
}

#[cube(launch_unchecked)]
fn im2col_kernel<F: Float>(
    image: &Tensor<F>,
    columns: &mut Tensor<F>,
    args: &Im2ColArgs,
    #[comptime] kernel_w_unroll: Option<u32>,
    #[comptime] has_padding: bool,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let batch_size = image.shape(0);
    let height = image.shape(2);
    let width = image.shape(3);

    let out_h = args.out_h;
    let out_w = args.out_w;

    if ABSOLUTE_POS > args.num_elements {
        return;
    }

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = ABSOLUTE_POS / out_w % out_h;
    let batch = ABSOLUTE_POS / (out_w * out_h) % batch_size;
    let channel = ABSOLUTE_POS / (out_w * out_h * batch_size) % image.shape(1);

    let kernel_w = kernel_w_unroll.unwrap_or(args.kernel_w);
    let unroll_w = kernel_w_unroll.is_some();

    let image_idx = batch * image.stride(0) + channel * image.stride(1);
    let col_idx = channel * args.kernel_h * kernel_w * args.col_size_1
        + batch * out_h * out_w
        + out_y * out_w
        + out_x;

    for kernel_y in 0..args.kernel_h {
        #[unroll(unroll_w)]
        for kernel_x in 0..kernel_w {
            let kernel_pos = kernel_y * kernel_w + kernel_x;
            let col_pos = col_idx + kernel_pos * args.col_size_1;

            if has_padding {
                let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32
                    - args.padding_h as i32;
                let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32
                    - args.padding_w as i32;
                if y >= 0 && x >= 0 && y < height as i32 && x < width as i32 {
                    let image_ptr = image_idx + y as u32 * width + x as u32;
                    columns[col_pos] = image[image_ptr];
                } else {
                    columns[col_pos] = F::new(0.0)
                };
            } else {
                let y = out_y * args.stride_h + kernel_y * args.dilation_h;
                let x = out_x * args.stride_w + kernel_x * args.dilation_w;
                let image_ptr = image_idx + y * image.stride(2) + x * image.stride(3);
                columns[col_pos] = image[image_ptr];
            }
        }
    }
}

#[allow(unused)]
// TODO check if better in cubecl or burn
pub fn batches_per_run(batch_size: usize, out_h: usize, out_w: usize) -> Option<usize> {
    #[cfg(test)]
    return Some(1);

    #[cfg(not(test))]
    {
        let cube_count_per_batch = (out_h * out_w).div_ceil(cubecl::PLANE_DIM_APPROX);
        let max_cube_count = u16::MAX as usize;
        let max_simultaneous = (max_cube_count / cube_count_per_batch).min(batch_size);
        if max_simultaneous == 0 {
            return None;
        }
        Some(
            (0..=max_simultaneous)
                .rev()
                .find(|per_run| batch_size % per_run == 0)
                .expect("Logically not possible"),
        )
    }
}

pub fn im2col<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    columns: TensorHandleRef<R>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: ConvOptions<2>,
) {
    let input: TensorHandle<R, E> = into_contiguous(client, &input);
    let [batch_size, in_channels, _, _] = input
        .shape
        .clone()
        .try_into()
        .expect("Input shape should have 4 dimensions");

    let col_shape_1 = batch_size * out_h * out_w;
    let num_elems = in_channels * batch_size * out_h * out_w;
    let kernel_w_unroll = (kernel_w <= 8).then_some(kernel_w as u32);
    let line_size = 1;

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        im2col_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            input.as_arg(line_size),
            columns.as_tensor_arg(line_size),
            Im2ColArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.padding[0] as u32),
                ScalarArg::new(options.padding[1] as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(out_h as u32),
                ScalarArg::new(out_w as u32),
                ScalarArg::new(col_shape_1 as u32),
                ScalarArg::new(num_elems as u32),
            ),
            kernel_w_unroll,
            options.padding != [0, 0],
        )
    };
}
