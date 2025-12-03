//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, tensor_line_size_parallel};

use cubecl_std::tensor::{
    MatrixBatchLayout, View, launch::ViewArg, layout::Coords3d, matrix_batch_layout,
};

use crate::{
    MatmulInputHandle, MatmulInputHandleRef,
    components::{
        MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError, MatrixLayout,
        global::memory::{GlobalLayout, GlobalLayoutConfig, GlobalLayoutLaunch, GlobalScaleLayout},
    },
};

#[cube]
fn load_unrolled<I: Numeric>(
    view: &View<Line<I>, Coords3d>,
    pos: Coords3d,
    #[comptime] layout: MatrixLayout,
    #[comptime] line_size: u32,
) -> Line<I> {
    comptime![assert!(line_size >= view.line_size())];
    let view_line_size = view.line_size();
    if comptime![view.line_size() == line_size] {
        view[pos]
    } else {
        let (b, row, col) = pos;
        let mut out = Line::empty(line_size);
        #[unroll]
        for i in range_stepped(0, line_size, view_line_size) {
            let pos = match layout {
                MatrixLayout::RowMajor => (b, row, col + i),
                MatrixLayout::ColMajor => (b, row + i, col),
            };
            let value = view[pos];
            #[unroll]
            for n in 0..view_line_size {
                out[i + n] = value[n];
            }
        }
        out
    }
}

#[cube(launch_unchecked)]
fn matmul_kernel<I: Numeric, M: Numeric, O: Numeric>(
    lhs: &View<Line<I>, Coords3d>,
    rhs: &View<Line<I>, Coords3d>,
    out: &mut Tensor<O>,
    #[define(I)] _input_dtype: StorageType,
    #[define(M)] _acc_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
) {
    let rank = out.rank();

    let (_, _, k) = lhs.shape();
    let size_m = out.shape(rank - 2);
    let size_n = out.shape(rank - 1);

    let batch = ABSOLUTE_POS_Z;
    let m = ABSOLUTE_POS_X;
    let n = ABSOLUTE_POS_Y;

    if m >= size_m || n >= size_n {
        terminate!();
    }

    let offset_out = batch * out.stride(rank - 2) * out.shape(rank - 2);

    let line_size = comptime![Ord::max(lhs.line_size(), rhs.line_size())];
    let mut sum = Line::empty(line_size).fill(O::from_int(0));

    for k in range_stepped(0, k, line_size) {
        let lhs = load_unrolled(lhs, (batch, m, k), MatrixLayout::RowMajor, line_size);
        let rhs = load_unrolled(rhs, (batch, k, n), MatrixLayout::ColMajor, line_size);

        sum += Line::cast_from(Line::<M>::cast_from(lhs) * Line::<M>::cast_from(rhs));
    }

    let mut out_index = m * out.stride(rank - 2) + n;
    out_index += offset_out;

    let unroll_sum = line_size != 1;
    if unroll_sum {
        let mut accum = O::from_int(0);
        // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
        // use SIMD instructions to speed up the computation
        #[unroll]
        for v in 0..line_size {
            accum += sum[v];
        }

        out[out_index] = accum;
    } else {
        out[out_index] = sum[0];
    }
}

/// Matrix multiplication using memory coalescing algorithm with custom cube dimensions
#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: MatmulInputHandle<R>,
    rhs: MatmulInputHandle<R>,
    out: &TensorHandleRef<'_, R>,
    dtypes: MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_ref(client, &lhs.as_ref(), &rhs.as_ref(), out, &dtypes)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let (cube_dim_x, cube_dim_y) = (32, 8);
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let lhs_layout = matrix_batch_layout(lhs.data().strides);
    let rhs_layout = matrix_batch_layout(rhs.data().strides);

    let lhs = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        lhs.into_contiguous(client)?
    } else {
        MatmulInputHandle::from_ref(lhs)
    };
    let lhs = lhs.as_ref();
    let rhs = MatmulInputHandle::from_ref(rhs);

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: MatmulInputHandle<R>| {
        rhs.swap_dims(dim1, dim2);
        let mut rhs = rhs.as_ref().into_contiguous(client)?;

        rhs.swap_dims(dim1, dim2);
        let returned: Result<MatmulInputHandle<R>, LaunchError> = Ok(rhs);
        returned
    };

    let rhs = match rhs_layout {
        MatrixBatchLayout::Contiguous => correct_rhs_layout(rhs)?,
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                rhs
            } else {
                correct_rhs_layout(rhs)?
            }
        }
        MatrixBatchLayout::HighlyPermuted => correct_rhs_layout(rhs)?,
    };
    let rhs = rhs.as_ref();

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = out.shape;

    let cube_count = simple_cube_count(lhs_shape, rhs_shape, out_shape, cube_dim_x, cube_dim_y)?;

    let lhs_line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtypes.lhs_global),
        lhs.data().shape,
        lhs.data().strides,
        rank - 1,
    );
    let rhs_line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtypes.rhs_global),
        rhs.data().shape,
        rhs.data().strides,
        rank - 2,
    );

    let problem = MatmulProblem {
        m: out_shape[rank - 2],
        n: out_shape[rank - 1],
        k: lhs_shape[rank - 1],
        lhs_batches: lhs_shape[..rank - 2].to_vec(),
        rhs_batches: rhs_shape[..rank - 2].to_vec(),
        out_batches: out_shape[..rank - 2].to_vec(),
        lhs_strides: lhs.data().strides.to_vec(),
        rhs_strides: rhs.data().strides.to_vec(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
    };

    fn view<'a, R: Runtime>(
        client: &ComputeClient<R>,
        handle: &'a MatmulInputHandleRef<'a, R>,
        layout: MatrixLayout,
        line_size: u8,
        problem: &MatmulProblem,
    ) -> ViewArg<'a, Coords3d, R> {
        // Checks off, other properties are unused
        let config = GlobalLayoutConfig {
            matrix_layout: layout,
            ..Default::default()
        };
        match handle {
            MatmulInputHandleRef::Normal(handle, _dtype) => {
                let layout = GlobalLayoutLaunch::from_handle_batched(
                    client, handle, problem, line_size, config,
                );
                ViewArg::new::<GlobalLayout>(handle.as_array_arg(line_size), layout)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
                ..
            } => {
                let (data_layout, scales_layout) = GlobalLayoutLaunch::from_quantized_handle(
                    client, data, scale, shape, problem, **scheme, line_size, config,
                );
                let data_view =
                    ViewArg::new::<GlobalLayout>(data.as_array_arg(line_size), data_layout);
                let scales_view =
                    ViewArg::new::<GlobalScaleLayout>(scale.as_array_arg(1), scales_layout);
                ViewArg::new_quantized(data_view, scales_view, **scheme)
            }
        }
    }

    let lhs_view = view(
        client,
        &lhs,
        MatrixLayout::RowMajor,
        lhs_line_size,
        &problem,
    );
    let rhs_view = view(
        client,
        &rhs,
        MatrixLayout::ColMajor,
        rhs_line_size,
        &problem,
    );

    let result = unsafe {
        matmul_kernel::launch_unchecked(
            client,
            cube_count,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            lhs_view,
            rhs_view,
            out.as_tensor_arg(1),
            *dtypes.lhs_global,
            *dtypes.acc_register,
            *dtypes.acc_global,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(MatmulSetupError::Launch(err)),
    }
}

#[allow(clippy::result_large_err)]
fn simple_cube_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> Result<CubeCount, MatmulSetupError> {
    let ndims = lhs_shape.len();
    let num_rows = lhs_shape[ndims - 2];
    let num_cols = rhs_shape[ndims - 1];

    let cubes_x = f32::ceil(num_rows as f32 / cube_dim_x as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / cube_dim_y as f32) as u32;
    let mut num_iter = 1u32;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        num_iter *= output_shape[i] as u32;
    }

    let result = CubeCount::Static(cubes_x, cubes_y, num_iter);
    let max_cube_count = u16::MAX as u32;

    if cubes_x > max_cube_count || cubes_y > max_cube_count || num_iter > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(result),
        ));
    }

    Ok(result)
}
