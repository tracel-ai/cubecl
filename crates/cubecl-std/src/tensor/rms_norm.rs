//! Root-mean-square (RMS) normalization kernels.
//!
//! The implementation normalizes across the last dimension of the input tensor and expects that
//! dimension to be contiguous. The kernels are vectorized whenever the runtime allows it and can
//! optionally fuse a bias addition after the scaling step.

use core::{any::TypeId, cmp, convert::TryFrom};

use cubecl::frontend::{Recip, Sqrt};
use cubecl::prelude::*;
use cubecl::tensor_line_size_parallel;
use cubecl_core as cubecl;
use cubecl_runtime::features::Plane;

use super::TensorHandle;

const MAX_LINES_PER_THREAD: u32 = 64;
const MAX_VECTOR_ELEMENTS_PER_THREAD: u32 = 1024;
const MAX_SUBGROUPS_PER_ROW: u32 = 32;
const LINES_PER_LANE_TARGET: u32 = 4;
const DEBUG_ASSERTIONS_ENABLED: bool = cfg!(debug_assertions);

const fn max_cached_lines_per_thread_for_line_size(line_size: u32) -> u32 {
    if line_size == 0 {
        return 1;
    }
    let element_bound = MAX_VECTOR_ELEMENTS_PER_THREAD / line_size;
    let element_bound = if element_bound == 0 { 1 } else { element_bound };
    let capped = if MAX_LINES_PER_THREAD < element_bound {
        MAX_LINES_PER_THREAD
    } else {
        element_bound
    };
    if capped == 0 { 1 } else { capped }
}

#[inline(always)]
fn max_cached_lines_per_thread(line_size: u32) -> u32 {
    max_cached_lines_per_thread_for_line_size(line_size)
}

#[cube]
fn reduce_sum_with_shuffle(value: f32, subgroup_size: u32, lane_id: u32) -> f32 {
    if subgroup_size == 0 {
        value
    } else {
        let mut sum = value;
        let is_pow_two = (subgroup_size & (subgroup_size - 1)) == 0;
        if is_pow_two {
            let mut offset = subgroup_size >> 1;
            while offset > 0 {
                sum += plane_shuffle_xor(sum, offset);
                offset >>= 1;
            }
            sum
        } else {
            if lane_id == 0 {
                let mut idx = 1u32;
                while idx < subgroup_size {
                    sum += plane_shuffle(value, idx);
                    idx += 1;
                }
            }
            plane_broadcast(sum, 0)
        }
    }
}

#[cube(fast_math = FastMath::AllowReciprocal)]
fn fast_inverse_sqrt(value: f32) -> f32 {
    Recip::recip(Sqrt::sqrt(value))
}

#[cube]
fn compute_inv_rms(total_sum: f32, axis_size: f32, eps: f32, allow_native_rsqrt: bool) -> f32 {
    let mean = total_sum / axis_size;
    let denom = mean + eps;
    if allow_native_rsqrt {
        fast_inverse_sqrt(denom)
    } else {
        1.0f32 / Sqrt::sqrt(denom)
    }
}

#[cube(launch_unchecked)]
fn rms_norm_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    weight: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    _num_rows: u32,
    lines_per_row: u32,
    axis_size: u32,
    eps: f32,
    use_fast_rsqrt: u32,
) {
    let row = CUBE_POS_X;

    let subgroup_size = PLANE_DIM;
    let subgroups_per_row = CUBE_DIM_X / subgroup_size;
    let active_threads = subgroups_per_row * subgroup_size;

    let line_size = comptime!(input.line_size());
    let lane_id = UNIT_POS_PLANE;
    let thread_linear = UNIT_POS_X;
    let subgroup_id = thread_linear / subgroup_size;
    let is_active_lane = subgroup_id < subgroups_per_row;

    let row_start = row * lines_per_row;

    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);
    let mut weight_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;
    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];
            // Weight vectors are [axis]-shaped, share the runtime vectorization, and are indexed by the line offset.
            let gamma = weight[line_index];

            value_cache[local_count] = values;
            weight_cache[local_count] = gamma;

            #[unroll]
            for lane in 0..line_size {
                let v = f32::cast_from(values[lane]);
                partial_sum += v * v;
            }

            local_count += 1;
            line_index += active_threads;
        }
    }

    let subgroup_sum = reduce_sum_with_shuffle(partial_sum, subgroup_size, lane_id);
    let axis = axis_size as f32;

    let inv_rms = if subgroups_per_row == 1u32 {
        let mut inv = 0.0f32;
        if lane_id == 0 {
            inv = compute_inv_rms(subgroup_sum, axis, eps, allow_native_rsqrt);
        }
        plane_broadcast(inv, 0)
    } else {
        let mut shared_partials = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW);
        if is_active_lane && lane_id == 0 {
            shared_partials[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += shared_partials[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                shared_partials[0] = reduced;
            }
        }
        sync_cube();

        let total_sum = shared_partials[0];
        let mut shared_inv = SharedMemory::<f32>::new(1u32);
        if subgroup_id == 0 && lane_id == 0 {
            shared_inv[0] = compute_inv_rms(total_sum, axis, eps, allow_native_rsqrt);
        }
        sync_cube();
        shared_inv[0]
    };

    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let gamma = weight_cache[iteration];
                let mut normalized = Line::<F>::empty(line_size);

                #[unroll]
                for lane in 0..line_size {
                    let v = f32::cast_from(values[lane]);
                    let g = f32::cast_from(gamma[lane]);
                    let result = v * inv_rms * g;
                    normalized[lane] = F::cast_from(result);
                }

                output[global_index] = normalized;
            }
            iteration += 1;
        }
    }
}

#[cube(launch_unchecked)]
fn rms_norm_bias_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    weight: &Tensor<Line<F>>,
    bias: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    _num_rows: u32,
    lines_per_row: u32,
    axis_size: u32,
    eps: f32,
    use_fast_rsqrt: u32,
) {
    let row = CUBE_POS_X;

    let subgroup_size = PLANE_DIM;
    let subgroups_per_row = CUBE_DIM_X / subgroup_size;
    let active_threads = subgroups_per_row * subgroup_size;

    let line_size = comptime!(input.line_size());
    let lane_id = UNIT_POS_PLANE;
    let thread_linear = UNIT_POS_X;
    let subgroup_id = thread_linear / subgroup_size;
    let is_active_lane = subgroup_id < subgroups_per_row;

    let row_start = row * lines_per_row;

    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);
    let mut weight_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);
    let mut bias_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;

    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];
            // Weight and bias vectors mirror the normalized axis/vectorization and use the same line offset indexing.
            let gamma = weight[line_index];
            let bias_line = bias[line_index];

            value_cache[local_count] = values;
            weight_cache[local_count] = gamma;
            bias_cache[local_count] = bias_line;

            #[unroll]
            for lane in 0..line_size {
                let v = f32::cast_from(values[lane]);
                partial_sum += v * v;
            }

            local_count += 1;
            line_index += active_threads;
        }
    }

    let subgroup_sum = reduce_sum_with_shuffle(partial_sum, subgroup_size, lane_id);
    let axis = axis_size as f32;

    let inv_rms = if subgroups_per_row == 1u32 {
        let mut inv = 0.0f32;
        if lane_id == 0 {
            inv = compute_inv_rms(subgroup_sum, axis, eps, allow_native_rsqrt);
        }
        plane_broadcast(inv, 0)
    } else {
        let mut shared_partials = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW);
        if is_active_lane && lane_id == 0 {
            shared_partials[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += shared_partials[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                shared_partials[0] = reduced;
            }
        }
        sync_cube();

        let total_sum = shared_partials[0];
        let mut shared_inv = SharedMemory::<f32>::new(1u32);
        if subgroup_id == 0 && lane_id == 0 {
            shared_inv[0] = compute_inv_rms(total_sum, axis, eps, allow_native_rsqrt);
        }
        sync_cube();
        shared_inv[0]
    };

    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let gamma = weight_cache[iteration];
                let bias_line = bias_cache[iteration];
                let mut normalized = Line::<F>::empty(line_size);

                #[unroll]
                for lane in 0..line_size {
                    let v = f32::cast_from(values[lane]);
                    let g = f32::cast_from(gamma[lane]);
                    let b = f32::cast_from(bias_line[lane]);
                    let result = v * inv_rms * g + b;
                    normalized[lane] = F::cast_from(result);
                }

                output[global_index] = normalized;
            }
            iteration += 1;
        }
    }
}

/// Launch RMS normalization and write the result into an existing output tensor.
///
/// The last dimension of `input`/`output` is normalized. `weight` (and `bias` if provided) must be
/// one-dimensional tensors whose length matches the normalized dimension.
pub fn launch<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandle<R, F>,
    weight: &TensorHandle<R, F>,
    bias: Option<&TensorHandle<R, F>>,
    output: &TensorHandle<R, F>,
    epsilon: f32,
) {
    launch_ref::<R, F>(
        client,
        input.as_ref(),
        weight.as_ref(),
        bias.map(|b| b.as_ref()),
        output.as_ref(),
        epsilon,
    );
}

/// Launch RMS normalization and allocate a new output tensor to store the result.
pub fn launch_alloc<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server>,
    input: &TensorHandle<R, F>,
    weight: &TensorHandle<R, F>,
    bias: Option<&TensorHandle<R, F>>,
    epsilon: f32,
) -> TensorHandle<R, F> {
    let output = TensorHandle::<R, F>::empty(client, input.shape.clone());
    launch_ref::<R, F>(
        client,
        input.as_ref(),
        weight.as_ref(),
        bias.map(|b| b.as_ref()),
        output.as_ref(),
        epsilon,
    );
    output
}

/// Launch RMS normalization using tensor handle references.
pub fn launch_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server>,
    input: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: Option<TensorHandleRef<R>>,
    output: TensorHandleRef<R>,
    epsilon: f32,
) {
    assert_eq!(
        input.shape, output.shape,
        "Input and output tensors must share the same shape"
    );

    let rank = input.shape.len();
    assert!(
        rank >= 1,
        "RMSNorm expects tensors with at least one dimension"
    );
    let axis = rank - 1;

    assert_eq!(
        input.strides[axis], 1,
        "The normalized dimension must be contiguous in memory"
    );
    assert_eq!(
        output.strides[axis], 1,
        "The output tensor must be contiguous along the normalized dimension"
    );

    assert_eq!(
        weight.shape.len(),
        1,
        "Weight tensor must be one dimensional"
    );
    assert_eq!(
        weight.shape[0], input.shape[axis],
        "Weight length must match the normalized dimension"
    );
    assert_eq!(weight.strides[0], 1, "Weight tensor must be contiguous");

    if let Some(bias_ref) = bias {
        assert_eq!(
            bias_ref.shape.len(),
            1,
            "Bias tensor must be one dimensional"
        );
        assert_eq!(
            bias_ref.shape[0], input.shape[axis],
            "Bias length must match the normalized dimension"
        );
        assert_eq!(bias_ref.strides[0], 1, "Bias tensor must be contiguous");
    }

    let axis_size = input.shape[axis];
    if axis_size == 0 {
        return;
    }

    let vectorization = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        axis,
    );
    let weight_vectorization = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        weight.shape,
        weight.strides,
        0,
    );
    assert_eq!(
        vectorization, weight_vectorization,
        "Weight tensor must use the same vectorization as the input"
    );

    if let Some(bias_ref) = bias {
        let bias_vectorization = tensor_line_size_parallel(
            R::supported_line_sizes().iter().cloned(),
            bias_ref.shape,
            bias_ref.strides,
            0,
        );
        assert_eq!(
            vectorization, bias_vectorization,
            "Bias tensor must use the same vectorization as the input"
        );
    }

    let line_size = vectorization as u32;
    let max_cached_lines = max_cached_lines_per_thread(line_size);
    let axis_size_u32 = u32::try_from(axis_size).expect("Axis size exceeds u32 range");
    assert_eq!(
        axis_size_u32 % line_size,
        0,
        "Normalized dimension must align with runtime vectorization width",
    );
    let lines_per_row = axis_size_u32 / line_size;

    let total_elements: usize = input.shape.iter().product();
    let num_rows = total_elements / axis_size;
    let num_rows_u32 = u32::try_from(num_rows).expect("Number of rows exceeds u32 range");

    let props = client.properties();
    let subgroup_size = cmp::max(props.hardware.plane_size_min, 1);
    let max_threads_axis = cmp::max(
        subgroup_size,
        cmp::min(
            props.hardware.max_cube_dim.x,
            props.hardware.max_units_per_cube,
        ),
    );
    let mut max_subgroups_hw = max_threads_axis / subgroup_size;
    if max_subgroups_hw == 0 {
        max_subgroups_hw = 1;
    }
    max_subgroups_hw = cmp::min(max_subgroups_hw, MAX_SUBGROUPS_PER_ROW);

    let mut subgroups_per_row = lines_per_row
        .div_ceil(LINES_PER_LANE_TARGET)
        .clamp(1, max_subgroups_hw);
    let mut threads_per_row = subgroups_per_row * subgroup_size;
    let mut per_thread_lines = lines_per_row.div_ceil(threads_per_row);

    while per_thread_lines > max_cached_lines && subgroups_per_row < max_subgroups_hw {
        subgroups_per_row += 1;
        threads_per_row = subgroups_per_row * subgroup_size;
        per_thread_lines = lines_per_row.div_ceil(threads_per_row);
    }

    assert!(
        subgroups_per_row > 0,
        "Invalid launch configuration: zero subgroups",
    );
    debug_assert!(
        per_thread_lines <= max_cached_lines,
        "launch config assigns {} lines per thread but cache only holds {}",
        per_thread_lines,
        max_cached_lines
    );
    assert!(
        per_thread_lines <= max_cached_lines,
        "RMSNorm configuration exceeds register allocation per lane",
    );
    let cube_dim = CubeDim::new_1d(threads_per_row);
    let cube_count = CubeCount::new_1d(num_rows_u32);

    let num_rows_arg = ScalarArg::new(num_rows_u32);
    let lines_per_row_arg = ScalarArg::new(lines_per_row);
    let axis_size_arg = ScalarArg::new(axis_size_u32);
    let eps_arg = ScalarArg::new(epsilon);
    let allow_native_rsqrt =
        props.features.plane.contains(Plane::Ops) && props.hardware.plane_size_min > 1;
    let use_fast_rsqrt = allow_native_rsqrt && TypeId::of::<F>() == TypeId::of::<f32>();
    let use_fast_rsqrt_arg = ScalarArg::new(if use_fast_rsqrt { 1u32 } else { 0u32 });

    unsafe {
        let input_arg =
            TensorArg::from_raw_parts::<F>(input.handle, input.strides, input.shape, vectorization);
        let weight_arg = TensorArg::from_raw_parts::<F>(
            weight.handle,
            weight.strides,
            weight.shape,
            vectorization,
        );
        let output_arg = TensorArg::from_raw_parts::<F>(
            output.handle,
            output.strides,
            output.shape,
            vectorization,
        );

        if let Some(bias_ref) = bias {
            let bias_arg = TensorArg::from_raw_parts::<F>(
                bias_ref.handle,
                bias_ref.strides,
                bias_ref.shape,
                vectorization,
            );

            rms_norm_bias_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                input_arg,
                weight_arg,
                bias_arg,
                output_arg,
                num_rows_arg,
                lines_per_row_arg,
                axis_size_arg,
                eps_arg,
                use_fast_rsqrt_arg,
            );
        } else {
            rms_norm_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                input_arg,
                weight_arg,
                output_arg,
                num_rows_arg,
                lines_per_row_arg,
                axis_size_arg,
                eps_arg,
                use_fast_rsqrt_arg,
            );
        }
    }
}
