use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise};
use cubecl_linalg::tensor::{into_contiguous, TensorHandle};

use crate::ConvTransposeOptions;

#[allow(clippy::too_many_arguments)]
pub fn col2im<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    columns: TensorHandleRef<R>,
    bias: Option<TensorHandleRef<R>>,
    out: TensorHandleRef<R>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: ConvTransposeOptions<2>,
) {
    let [_, col_size_1] = columns
        .shape
        .try_into()
        .expect("Columns shape should have 2 dimensions");

    let columns: TensorHandle<R, E> = into_contiguous(client, &columns);
    let has_bias = bias.is_some();
    let bias = bias
        .map(|bias| into_contiguous(client, &bias))
        .unwrap_or_else(|| TensorHandle::<R, E>::empty(client, vec![1]));

    let vectorization = 1;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(out.size(), cube_dim);

    unsafe {
        col2im_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            columns.as_arg(vectorization),
            bias.as_arg(vectorization),
            out.as_tensor_arg(vectorization),
            Col2ImArgsLaunch::new(
                ScalarArg::new(out_h as u32),
                ScalarArg::new(out_w as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(options.padding[0] as u32),
                ScalarArg::new(options.padding[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(col_size_1 as u32),
            ),
            has_bias,
        )
    };
}

#[derive(CubeLaunch)]
struct Col2ImArgs {
    out_h: u32,
    out_w: u32,

    kernel_h: u32,
    kernel_w: u32,

    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    stride_h: u32,
    stride_w: u32,

    col_size_1: u32,
}

#[cube(launch_unchecked)]
fn col2im_kernel<F: Float>(
    columns: &Tensor<F>,
    bias: &Tensor<F>,
    image: &mut Tensor<F>,
    args: &Col2ImArgs,
    #[comptime] has_bias: bool,
) {
    if ABSOLUTE_POS >= image.len() {
        return;
    }

    let im_x = ABSOLUTE_POS % image.shape(3) + args.pad_w;
    let im_y = ABSOLUTE_POS / image.stride(2) % image.shape(2) + args.pad_h;
    let ch_im = ABSOLUTE_POS / image.stride(1) % image.shape(1);
    let batch = ABSOLUTE_POS / image.stride(0);

    let kernel_extent_w = (args.kernel_w - 1) * args.dilation_w + 1;
    let kernel_extent_h = (args.kernel_h - 1) * args.dilation_h + 1;

    let mut val = F::new(0.0);

    let x_col_start = if im_x >= kernel_extent_w {
        (im_x - kernel_extent_w) / args.stride_w + 1
    } else {
        0u32
    };
    let x_col_end = Min::min(im_x / args.stride_w + 1, args.out_w);
    let y_col_start = if im_y >= kernel_extent_h {
        (im_y - kernel_extent_h) / args.stride_h + 1
    } else {
        0u32
    };
    let y_col_end = Min::min(im_y / args.stride_h + 1, args.out_h);

    for col_y in y_col_start..y_col_end {
        let kernel_y = im_y - col_y * args.stride_h;
        for col_x in x_col_start..x_col_end {
            let kernel_x = im_x - col_x * args.stride_w;

            if kernel_y % args.dilation_h == 0 && kernel_x % args.dilation_w == 0 {
                let kernel_y = kernel_y / args.dilation_h;
                let kernel_x = kernel_x / args.dilation_w;

                let col_pos = ch_im * args.kernel_h * args.kernel_w * args.col_size_1
                    + kernel_y * args.kernel_w * args.col_size_1
                    + kernel_x * args.col_size_1
                    + batch * args.out_h * args.out_w
                    + col_y * args.out_w
                    + col_x;
                val += columns[col_pos];
            }
        }
    }

    if has_bias {
        image[ABSOLUTE_POS] = val + bias[ch_im];
    } else {
        image[ABSOLUTE_POS] = val;
    }
}
