//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    ir::{ElemType, IntKind, UIntKind},
};

use cubecl_std::tensor::{
    MatrixBatchLayout, TensorHandle, View, into_contiguous, launch::ViewArg, layout::Coords3d,
    matrix_batch_layout,
};

use crate::{
    MatmulInputHandle, MatmulInputHandleRef,
    components::{
        MatmulAvailabilityError, MatmulProblem, MatmulSetupError, MatrixLayout,
        global::memory::{
            BatchedGlobalLayout, BatchedGlobalLayoutLaunch, BatchedGlobalScaleLayout,
            GlobalMemoryConfig,
        },
    },
};

#[cube(launch_unchecked)]
fn matmul_kernel<I: Numeric, M: Numeric, O: Numeric>(
    lhs: &View<Line<I>, Coords3d>,
    rhs: &View<Line<I>, Coords3d>,
    out: &mut Tensor<O>,
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

    let line_size = lhs.line_size();

    let offset_out = batch * out.stride(rank - 2) * out.shape(rank - 2);

    let mut sum = Line::empty(line_size).fill(O::from_int(0));

    for k in range_stepped(0, k, line_size) {
        sum += Line::cast_from(
            Line::<M>::cast_from(lhs[(batch, m, k)]) * Line::<M>::cast_from(rhs[(batch, k, n)]),
        );
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
pub fn launch<R: Runtime, EI: Numeric, EO: Numeric>(
    client: &ComputeClient<R::Server>,
    lhs: MatmulInputHandle<R, EI>,
    rhs: MatmulInputHandle<R, EI>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulSetupError> {
    launch_ref::<R, EI, EO>(client, &lhs.as_ref(), &rhs.as_ref(), out)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, EI: Numeric, EO: Numeric>(
    client: &ComputeClient<R::Server>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulSetupError> {
    let (cube_dim_x, cube_dim_y) = (32, 8);
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let mut lhs = *lhs;
    let mut rhs = *rhs;

    let lhs_layout = matrix_batch_layout(lhs.data().strides);
    let rhs_layout = matrix_batch_layout(rhs.data().strides);

    let lhs_data = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        into_contiguous::<R, EI>(client, lhs.data())
    } else {
        TensorHandle::from_ref(lhs.data())
    };
    let rhs_data = TensorHandle::from_ref(rhs.data());

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: TensorHandle<R, EI>| {
        let rhs_original_shape = rhs.shape.to_vec();
        rhs.strides.swap(dim1, dim2);
        rhs.shape.swap(dim1, dim2);

        let mut rhs = into_contiguous::<R, EI>(client, &rhs.as_ref());

        rhs.strides.swap(dim1, dim2);
        rhs.shape.swap(dim1, dim2);

        (rhs_original_shape, rhs)
    };

    let (rhs_original_shape, rhs_data) = match rhs_layout {
        MatrixBatchLayout::Contiguous => correct_rhs_layout(rhs_data),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                let rhs_original_shape = rhs_data.shape.to_vec();
                (rhs_original_shape, rhs_data)
            } else {
                correct_rhs_layout(rhs_data)
            }
        }
        MatrixBatchLayout::HighlyPermuted => correct_rhs_layout(rhs_data),
    };

    let cube_count = simple_cube_count(
        &lhs_data.shape,
        &rhs_original_shape,
        out.shape,
        cube_dim_x,
        cube_dim_y,
    )?;

    let line_size = match lhs_data.shape[rank - 1] % 4 == 0 {
        true => 4,
        false => 1,
    };
    let line_size = if lhs.scale().is_some() || rhs.scale().is_some() {
        1
    } else {
        line_size
    };

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = out.shape;
    let problem = MatmulProblem {
        m: out_shape[rank - 2],
        n: out_shape[rank - 1],
        k: lhs_shape[rank - 1],
        lhs_batches: lhs_shape[..rank - 2].to_vec(),
        rhs_batches: rhs_shape[..rank - 2].to_vec(),
        out_batches: out_shape[..rank - 2].to_vec(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
    };

    let launch = match EI::as_type_native_unchecked().elem_type() {
        ElemType::Int(IntKind::I8) => matmul_kernel::launch_unchecked::<EI, i16, EO, R>,
        ElemType::Int(IntKind::I16) | ElemType::UInt(UIntKind::U16) => {
            matmul_kernel::launch_unchecked::<EI, i32, EO, R>
        }
        ElemType::UInt(UIntKind::U8) => matmul_kernel::launch_unchecked::<EI, u16, EO, R>,
        _ => matmul_kernel::launch_unchecked::<EI, EI, EO, R>,
    };

    *lhs.data_mut() = lhs_data.as_ref();
    *rhs.data_mut() = rhs_data.as_ref();

    fn view<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        handle: &'a MatmulInputHandleRef<'a, R>,
        layout: MatrixLayout,
        line_size: u8,
        problem: &MatmulProblem,
    ) -> ViewArg<'a, Coords3d, R> {
        // Checks off, other properties are unused
        let mem_config = GlobalMemoryConfig {
            global_line_size: line_size as u32,
            matrix_layout: layout,
            ..Default::default()
        };
        match handle {
            MatmulInputHandleRef::Normal(handle) => {
                let layout =
                    BatchedGlobalLayoutLaunch::from_handle(client, handle, problem, mem_config);
                ViewArg::new::<BatchedGlobalLayout>(handle.as_array_arg(line_size), layout)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => {
                let (data_layout, scales_layout) = BatchedGlobalLayoutLaunch::from_quantized_handle(
                    client, data, scale, shape, problem, mem_config, **scheme,
                );
                let data_view =
                    ViewArg::new::<BatchedGlobalLayout>(data.as_array_arg(line_size), data_layout);
                let scales_view = ViewArg::new::<BatchedGlobalScaleLayout>(
                    scale.as_array_arg(line_size),
                    scales_layout,
                );
                ViewArg::new_quantized(data_view, scales_view, **scheme)
            }
        }
    }

    let lhs_view = view(client, &lhs, MatrixLayout::RowMajor, line_size, &problem);
    let rhs_view = view(client, &rhs, MatrixLayout::ColMajor, line_size, &problem);

    unsafe {
        launch(
            client,
            cube_count,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            lhs_view,
            rhs_view,
            out.as_tensor_arg(1),
        );
    };

    Ok(())
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
