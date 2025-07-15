//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::tensor::{MatrixBatchLayout, TensorHandle, into_contiguous, matrix_batch_layout};

use crate::components::{MatmulAvailabilityError, MatmulSetupError};

#[cube(launch_unchecked)]
fn matmul_kernel<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    out: &mut Tensor<N>,
    // number of dimensions not involved in the matmul
    #[comptime] num_batches: Option<u32>,
) {
    let rank = out.rank();
    let end = num_batches.unwrap_or_else(|| rank - 2);
    let unroll = num_batches.is_some();

    let n_rows = lhs.shape(rank - 2);
    let n_cols = rhs.shape(rank - 1);
    let mut k = rhs.shape(rank - 2);

    let batch_pos = ABSOLUTE_POS_Z;
    let row = CUBE_DIM_X * CUBE_POS_X + UNIT_POS_X;
    let col = CUBE_DIM_Y * CUBE_POS_Y + UNIT_POS_Y;

    if row >= n_rows || col >= n_cols {
        terminate!();
    }

    let line_size = lhs.line_size();

    let mut offset_lhs = 0;
    let mut offset_rhs = 0;
    let offset_out = batch_pos * out.stride(rank - 2) * out.shape(rank - 2);

    #[unroll(unroll)]
    for i in 0..end {
        let ogwl = offset_out / out.stride(i);

        offset_lhs += ogwl % lhs.shape(i) * lhs.stride(i);
        offset_rhs += ogwl % rhs.shape(i) * rhs.stride(i);
    }

    offset_lhs /= line_size.runtime();
    offset_rhs /= line_size.runtime();

    let mut sum = Line::empty(line_size).fill(N::from_int(0));

    k /= line_size.runtime();

    for i in 0..k {
        let lhs_index = row * lhs.stride(rank - 2) / line_size + i + offset_lhs;
        let rhs_index = col * rhs.stride(rank - 1) / line_size + i + offset_rhs;

        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    let mut out_index = row * out.stride(rank - 2) + col;
    out_index += offset_out;

    let unroll_sum = line_size != 1;
    if unroll_sum {
        let mut accum = N::from_int(0);
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
pub fn launch_ref<R: Runtime, E: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulSetupError> {
    let lhs = TensorHandle::<R, E>::from_ref(lhs);
    let rhs = TensorHandle::<R, E>::from_ref(rhs);

    launch(client, lhs, rhs, out)
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, E: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, E>,
    rhs: TensorHandle<R, E>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulSetupError> {
    let (cube_dim_x, cube_dim_y) = (32, 8);
    let ndims = lhs.shape.len();
    let dim1 = ndims - 1;
    let dim2 = ndims - 2;

    let lhs_layout = matrix_batch_layout(&lhs.strides);
    let rhs_layout = matrix_batch_layout(&rhs.strides);

    let lhs = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        into_contiguous::<R, E>(client, &lhs.as_ref())
    } else {
        lhs
    };

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: TensorHandle<R, E>| {
        let rhs_original_shape = rhs.shape.to_vec();
        rhs.strides.swap(dim1, dim2);
        rhs.shape.swap(dim1, dim2);

        let mut rhs = into_contiguous::<R, E>(client, &rhs.as_ref());

        rhs.strides.swap(dim1, dim2);
        rhs.shape.swap(dim1, dim2);

        (rhs_original_shape, rhs)
    };

    let (rhs_original_shape, rhs) = match rhs_layout {
        MatrixBatchLayout::Contiguous => correct_rhs_layout(rhs),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                let rhs_original_shape = rhs.shape.to_vec();
                (rhs_original_shape, rhs)
            } else {
                correct_rhs_layout(rhs)
            }
        }
        MatrixBatchLayout::HighlyPermuted => correct_rhs_layout(rhs),
    };

    let cube_count = simple_cube_count(
        &lhs.shape,
        &rhs_original_shape,
        out.shape,
        cube_dim_x,
        cube_dim_y,
    )?;

    let vectorization_factor = match lhs.shape[ndims - 1] % 4 == 0 {
        true => 4,
        false => 1,
    };

    unsafe {
        matmul_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            lhs.as_arg(vectorization_factor),
            rhs.as_arg(vectorization_factor),
            out.as_tensor_arg(1),
            Some(ndims as u32 - 2),
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
