//! RMSNorm kernels - Optimized for memory bandwidth
//!
//! This implementation provides two kernel variants:
//! - V1 (streaming): Minimal register usage, re-reads weights (default)
//! - V2 (smem): Stages weights/bias in shared memory (currently disabled)
//!
//! Key optimizations:
//! 1. Vectorization: 256-bit (vec=8) for F32, 128-bit (vec=4) for F16/BF16
//! 2. Occupancy enforcement: Maintains ≥512 threads for latency hiding
//! 3. Reduced register pressure: MAX_LINES_PER_THREAD=4 to avoid spills
//! 4. Fast math: Native rsqrt for inv_rms computation when available
//!
//! ## Performance Characteristics
//!
//! F32 performance is good (70-90% of theoretical bandwidth on modern GPUs).
//! F16/BF16 performance is currently hardware/driver dependent:
//! - Some GPUs: F16 approaches F32 performance (expected behavior)
//! - Others: F16 is 10-15× slower than F32 (driver/hardware limitation)
//!
//! Wide vectorization (vec=16) for F16 causes severe slowdowns on some hardware
//! despite maintaining full occupancy. Root cause is under investigation but
//! suspected to be driver-level issues with F16 memory transactions.
//!
//! ## Environment Variables
//!
//! - CUBECL_RMS_LOG=1: Enable diagnostic logging
//! - CUBECL_RMS_VARIANT=stream|smem: Force kernel variant
//! - CUBECL_RMS_VEC=4|8|16: Override vectorization width

use core::{cmp, convert::TryFrom};
use std::env;
use std::sync::{LazyLock, Mutex};
use std::collections::HashSet;

use cubecl_core as cubecl;
use cubecl::frontend::{Recip, Sqrt};
use cubecl::prelude::*;
use cubecl::tensor_line_size_parallel;
use cubecl_runtime::features::Plane;

use super::TensorHandle;

// Tuning constants
// CRITICAL: Keep this small (4) to avoid register spills! Was 64, caused 10-100x slowdowns.
const MAX_LINES_PER_THREAD: u32 = 4;
const MAX_VECTOR_ELEMENTS_PER_THREAD: u32 = 1024;
const MAX_SUBGROUPS_PER_ROW: u32 = 32;
const LINES_PER_LANE_TARGET_F32: u32 = 4;
const LINES_PER_LANE_TARGET_F16: u32 = 8; // More ILP for half types
const DEBUG_ASSERTIONS_ENABLED: bool = cfg!(debug_assertions);

// Shared memory capacity for V2 kernel (elements, not lines)
// Max for typical row: 4096 cols / 4 line_size = 1024 lines * 16 max_vec = 16384 elements
// But we need to be conservative for shared memory limits (~48KB)
// At 2 bytes per element (f16), 8192 elements = 16KB per buffer (weight+bias = 32KB total)
const MAX_SMEM_ELEMENTS_PER_BUFFER: u32 = 8192;

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

/// Get preferred vectorization for a dtype to maximize memory bandwidth
///
/// Strategy: Use wide vectorization for F16/BF16 to maximize bytes/thread
/// - F16/BF16 vec=16: 32 bytes/thread (256-bit loads) → maximize bandwidth
/// - F16/BF16 vec=8:  16 bytes/thread (128-bit loads) → fallback if vec=16 fails alignment
/// - F32 vec=8:       32 bytes/thread (256-bit loads) → maximize bandwidth
/// - F32 vec=4:       16 bytes/thread (128-bit loads) → fallback
///
/// The previous vec=4-for-all strategy caused catastrophic F16 performance:
/// - F16 vec=4: 8 bytes/thread → 27 GB/s (memory subsystem starved)
/// - F16 vec=16: 32 bytes/thread → ~600+ GB/s expected (20-30× faster)
///
/// Wide vectorization is the correct trade-off: fewer threads doing more work each
/// beats many threads doing tiny work (memory bandwidth >> occupancy for this kernel)
fn preferred_vectorization_for_dtype<F: Float>() -> Vec<u8> {
    let elem_size = core::mem::size_of::<F>();
    match elem_size {
        // TEMPORARILY REVERTING: Wide vectorization (vec=16) causes 12× slowdown on some hardware
        // Suspected issues: scalar memory transactions, driver bugs, or alignment problems
        // TODO: Investigate with PTX/SASS dumps and Nsight Compute
        2 => vec![4, 2, 1],  // F16/BF16: fallback to vec=4 (same bytes/thread as F32)
        4 => vec![8, 4, 2, 1],      // F32: prefer 256-bit (8 elems) or 128-bit (4 elems)
        8 => vec![4, 2, 1],         // F64: prefer 256-bit (4 elems)
        _ => vec![4, 2, 1],         // Fallback
    }
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

/// V1: Streaming kernel - minimal register usage, re-reads weights
/// Best for: huge rows, limited registers
#[cube(launch_unchecked)]
fn rms_norm_kernel_stream<F: Float>(
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

    // CRITICAL: MAX_LINES_PER_THREAD now set to 4 (was 64) to avoid register spills
    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;

    // Pass 1: Load input and compute sum of squares
    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];

            value_cache[local_count] = values;

            // Inline conversion (Test 2: removed load_line_as_f32 helper)
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

    // Reduce and compute inv_rms (2 sync barriers for multi-subgroup)
    let inv_rms = if subgroups_per_row == 1u32 {
        let mut inv = 0.0f32;
        if lane_id == 0 {
            inv = compute_inv_rms(subgroup_sum, axis, eps, allow_native_rsqrt);
        }
        plane_broadcast(inv, 0)
    } else {
        // Use extra slot to store final inv_rms, saves one sync barrier
        let mut shared_mem = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW + 1u32);
        if is_active_lane && lane_id == 0 {
            shared_mem[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += shared_mem[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                shared_mem[MAX_SUBGROUPS_PER_ROW] = compute_inv_rms(reduced, axis, eps, allow_native_rsqrt);
            }
        }
        sync_cube();
        shared_mem[MAX_SUBGROUPS_PER_ROW]
    };

    // Pass 2: Apply normalization (re-read weights to save registers)
    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let gamma = weight[line_offset]; // Re-read (small cost, saves registers)
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

/// V1: Streaming kernel with bias
#[cube(launch_unchecked)]
fn rms_norm_bias_kernel_stream<F: Float>(
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

    // CRITICAL: MAX_LINES_PER_THREAD now set to 4 (was 64) to avoid register spills
    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;

    // Pass 1: Compute sum of squares
    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];

            value_cache[local_count] = values;

            // Inline conversion (Test 2: removed load_line_as_f32 helper)
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
        let mut shared_mem = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW + 1u32);
        if is_active_lane && lane_id == 0 {
            shared_mem[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += shared_mem[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                shared_mem[MAX_SUBGROUPS_PER_ROW] = compute_inv_rms(reduced, axis, eps, allow_native_rsqrt);
            }
        }
        sync_cube();
        shared_mem[MAX_SUBGROUPS_PER_ROW]
    };

    // Pass 2: Apply normalization with bias
    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let gamma = weight[line_offset];
                let bias_line = bias[line_offset];
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

/// V2: Shared memory kernel without bias - stages weight in smem for better reuse
/// Best for: medium-large rows where weight reuse across passes benefits from smem
#[cube(launch_unchecked)]
fn rms_norm_kernel_smem<F: Float>(
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

    // Allocate shared memory for:
    // 1. Reduction intermediate values (MAX_SUBGROUPS_PER_ROW + 1)
    // 2. Weight staging (MAX_SMEM_ELEMENTS_PER_BUFFER)
    let mut reduction_smem = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW + 1u32);
    let mut weight_smem = SharedMemory::<F>::new(MAX_SMEM_ELEMENTS_PER_BUFFER);

    // Calculate actual elements needed
    let smem_elements = lines_per_row * line_size;

    // Bounds check - if row is too large, this kernel should not have been selected
    if smem_elements > MAX_SMEM_ELEMENTS_PER_BUFFER {
        if DEBUG_ASSERTIONS_ENABLED {
            terminate!();
        }
    }

    // Cooperative load of weight into shared memory
    let mut load_idx = thread_linear;
    while load_idx < lines_per_row {
        let w_line = weight[load_idx];
        let smem_base = load_idx * line_size;

        #[unroll]
        for lane in 0..line_size {
            weight_smem[smem_base + lane] = w_line[lane];
        }

        load_idx += active_threads;
    }
    sync_cube();

    // Cache input values
    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;

    // Pass 1: Compute sum of squares
    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];

            value_cache[local_count] = values;

            // Inline conversion (Test 2: removed load_line_as_f32 helper)
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
        if is_active_lane && lane_id == 0 {
            reduction_smem[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += reduction_smem[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                reduction_smem[MAX_SUBGROUPS_PER_ROW] = compute_inv_rms(reduced, axis, eps, allow_native_rsqrt);
            }
        }
        sync_cube();
        reduction_smem[MAX_SUBGROUPS_PER_ROW]
    };

    // Pass 2: Apply normalization using smem-staged weight
    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let smem_base = line_offset * line_size;
                let mut normalized = Line::<F>::empty(line_size);

                #[unroll]
                for lane in 0..line_size {
                    let v = f32::cast_from(values[lane]);
                    let g = f32::cast_from(weight_smem[smem_base + lane]);
                    let result = v * inv_rms * g;
                    normalized[lane] = F::cast_from(result);
                }

                output[global_index] = normalized;
            }
            iteration += 1;
        }
    }
}

/// V2: Shared memory kernel with bias - stages weight/bias in smem for better reuse
/// Best for: bias=true paths, medium-large rows
#[cube(launch_unchecked)]
fn rms_norm_bias_kernel_smem<F: Float>(
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

    // Allocate shared memory for:
    // 1. Reduction intermediate values (MAX_SUBGROUPS_PER_ROW + 1)
    // 2. Weight staging (MAX_SMEM_ELEMENTS_PER_BUFFER)
    // 3. Bias staging (MAX_SMEM_ELEMENTS_PER_BUFFER)
    let mut reduction_smem = SharedMemory::<f32>::new(MAX_SUBGROUPS_PER_ROW + 1u32);

    // Use fixed-size shared memory buffers
    let mut weight_smem = SharedMemory::<F>::new(MAX_SMEM_ELEMENTS_PER_BUFFER);
    let mut bias_smem = SharedMemory::<F>::new(MAX_SMEM_ELEMENTS_PER_BUFFER);

    // Calculate actual elements needed
    let smem_elements = lines_per_row * line_size;

    // Bounds check - if row is too large, this kernel should not have been selected
    if smem_elements > MAX_SMEM_ELEMENTS_PER_BUFFER {
        // This should never happen if heuristic is correct
        if DEBUG_ASSERTIONS_ENABLED {
            terminate!();
        }
    }

    // Cooperative load of weight and bias into shared memory
    // Each thread loads one or more lines, unpacking them into smem
    let mut load_idx = thread_linear;
    while load_idx < lines_per_row {
        let w_line = weight[load_idx];
        let b_line = bias[load_idx];
        let smem_base = load_idx * line_size;

        #[unroll]
        for lane in 0..line_size {
            weight_smem[smem_base + lane] = w_line[lane];
            bias_smem[smem_base + lane] = b_line[lane];
        }

        load_idx += active_threads;
    }
    sync_cube();

    // Cache input values
    let mut value_cache = Array::<Line<F>>::vectorized(MAX_LINES_PER_THREAD, line_size);

    let mut partial_sum = 0.0f32;
    let mut local_count = 0u32;
    let allow_native_rsqrt = use_fast_rsqrt != 0;

    // Pass 1: Compute sum of squares
    if is_active_lane {
        let mut line_index = thread_linear;
        while line_index < lines_per_row {
            if DEBUG_ASSERTIONS_ENABLED && local_count >= MAX_LINES_PER_THREAD {
                terminate!();
            }
            let global_index = row_start + line_index;
            let values = input[global_index];

            value_cache[local_count] = values;

            // Inline conversion (Test 2: removed load_line_as_f32 helper)
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
        if is_active_lane && lane_id == 0 {
            reduction_smem[subgroup_id] = subgroup_sum;
        }
        sync_cube();

        if subgroup_id == 0 {
            let mut accumulator = 0.0f32;
            let mut idx = lane_id;
            while idx < subgroups_per_row {
                accumulator += reduction_smem[idx];
                idx += subgroup_size;
            }
            let reduced = reduce_sum_with_shuffle(accumulator, subgroup_size, lane_id);
            if lane_id == 0 {
                reduction_smem[MAX_SUBGROUPS_PER_ROW] = compute_inv_rms(reduced, axis, eps, allow_native_rsqrt);
            }
        }
        sync_cube();
        reduction_smem[MAX_SUBGROUPS_PER_ROW]
    };

    // Pass 2: Apply normalization using smem-staged weight/bias
    if is_active_lane {
        let mut iteration = 0u32;
        while iteration < local_count {
            let line_offset = thread_linear + iteration * active_threads;
            if line_offset < lines_per_row {
                let global_index = row_start + line_offset;
                let values = value_cache[iteration];
                let smem_base = line_offset * line_size;
                let mut normalized = Line::<F>::empty(line_size);

                #[unroll]
                for lane in 0..line_size {
                    let v = f32::cast_from(values[lane]);
                    let g = f32::cast_from(weight_smem[smem_base + lane]);
                    let b = f32::cast_from(bias_smem[smem_base + lane]);
                    let result = v * inv_rms * g + b;
                    normalized[lane] = F::cast_from(result);
                }

                output[global_index] = normalized;
            }
            iteration += 1;
        }
    }
}

/// Heuristic: decide whether to use smem variant
///
/// V2 (smem staging) investigation results:
/// - F32: Reasonable performance, but no significant advantage over V1
/// - F16 with bias: CATASTROPHIC regression (12-100× slower than V1)
/// - F16 without bias: Still poor performance (no improvement)
///
/// Root causes under investigation:
/// 1. Shared memory bank conflicts with wide vectorization (vec=16)
/// 2. Cooperative load serialization during weight/bias staging
/// 3. Extra sync barriers (sync_cube) adding latency
/// 4. Memory access patterns not coalescing properly
///
/// Disabled V2 for all cases until issues are resolved.
/// Use CUBECL_RMS_VARIANT=smem to force V2 for debugging.
fn should_use_smem_variant(lines_per_row: u32, has_bias: bool, line_size: u32) -> bool {
    // V2 currently disabled due to severe performance regression
    // Keeping implementation for future investigation and optimization
    let _ = (lines_per_row, has_bias, line_size);
    false
}

/// Launch RMS normalization and write the result into an existing output tensor.
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

    // OPTIMIZATION: Prefer wider vectorization for f16/bf16, but limit f32 to avoid over-vectorization
    let preferred_vec = preferred_vectorization_for_dtype::<F>();
    let supported_line_sizes = R::supported_line_sizes();

    // Check for CUBECL_RMS_VEC environment override
    let vectorization = if let Ok(vec_override) = env::var("CUBECL_RMS_VEC") {
        if let Ok(forced_vec) = vec_override.parse::<u8>() {
            // Validate: must be supported and must divide the axis size
            if supported_line_sizes.contains(&forced_vec) && axis_size % (forced_vec as usize) == 0 {
                if env::var("CUBECL_RMS_LOG").is_ok() {
                    eprintln!("[RMSNorm] Forcing vectorization={} via CUBECL_RMS_VEC", forced_vec);
                }
                forced_vec
            } else {
                eprintln!(
                    "[RMSNorm] WARNING: CUBECL_RMS_VEC={} invalid (not supported or doesn't divide axis_size={}), using default",
                    forced_vec, axis_size
                );
                // Fall through to default calculation
                let allowed_sizes: Vec<u8> = preferred_vec.into_iter()
                    .filter(|s| supported_line_sizes.contains(s))
                    .collect();
                tensor_line_size_parallel(
                    allowed_sizes.iter().cloned(),
                    input.shape,
                    input.strides,
                    axis,
                )
            }
        } else {
            eprintln!("[RMSNorm] WARNING: CUBECL_RMS_VEC='{}' invalid, using default", vec_override);
            let allowed_sizes: Vec<u8> = preferred_vec.into_iter()
                .filter(|s| supported_line_sizes.contains(s))
                .collect();
            tensor_line_size_parallel(
                allowed_sizes.iter().cloned(),
                input.shape,
                input.strides,
                axis,
            )
        }
    } else {
        // Default: ONLY use preferred sizes that are supported - don't add extras!
        // This prevents f32 from using vec=16 when we only want vec=4
        let allowed_sizes: Vec<u8> = preferred_vec.into_iter()
            .filter(|s| supported_line_sizes.contains(s))
            .collect();
        tensor_line_size_parallel(
            allowed_sizes.iter().cloned(),
            input.shape,
            input.strides,
            axis,
        )
    };

    // For weight/bias, use the same vectorization as input (simpler - just validate they match)
    let weight_vectorization = vectorization;
    // Validate weight can use same vectorization
    assert_eq!(
        weight_vectorization,
        tensor_line_size_parallel(
            std::iter::once(vectorization),
            weight.shape,
            weight.strides,
            0,
        ),
        "Weight tensor must support the input's vectorization"
    );
    assert_eq!(
        vectorization, weight_vectorization,
        "Weight tensor must use the same vectorization as the input"
    );

    if let Some(bias_ref) = bias {
        let bias_vectorization = vectorization;
        // Validate bias can use same vectorization
        assert_eq!(
            bias_vectorization,
            tensor_line_size_parallel(
                std::iter::once(vectorization),
                bias_ref.shape,
                bias_ref.strides,
                0,
            ),
            "Bias tensor must support the input's vectorization"
        );
    }

    let line_size = vectorization as u32;
    let max_cached_lines = max_cached_lines_per_thread(line_size);
    let axis_size_u32 = u32::try_from(axis_size).expect("Axis size exceeds u32 range");

    // Diagnostic logging (only log once to avoid spam)
    static LOGGED_CONFIGS: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
    if env::var("CUBECL_RMS_LOG").is_ok() {
        let config_key = format!("{:?}_{}_{}_{}", input.shape, core::mem::size_of::<F>(), axis, vectorization);
        let mut logged = LOGGED_CONFIGS.lock().unwrap();
        if !logged.contains(&config_key) {
            eprintln!(
                "[RMSNorm] dtype_size={} shape={:?} axis={} vectorization={} (preferred: {:?})",
                core::mem::size_of::<F>(),
                input.shape,
                axis,
                vectorization,
                preferred_vectorization_for_dtype::<F>()
            );
            logged.insert(config_key.clone());
        }
    }
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

    // Use dtype-aware ILP target
    let elem_size = core::mem::size_of::<F>();
    let lines_per_lane_target = if elem_size == 2 {
        LINES_PER_LANE_TARGET_F16
    } else {
        LINES_PER_LANE_TARGET_F32
    };

    let mut subgroups_per_row = lines_per_row
        .div_ceil(lines_per_lane_target)
        .clamp(1, max_subgroups_hw);

    // CRITICAL: Maintain high occupancy even with wide vectorization
    // Wide vectorization (vec=16) reduces lines_per_row, but we need many threads
    // to hide memory latency. Enforce a minimum thread count for good occupancy.
    // Example: vec=16 → 256 lines, but launch 512-1024 threads (some threads idle, but GPU stays busy)
    const MIN_THREADS_FOR_OCCUPANCY: u32 = 512; // Target at least 50% occupancy
    let min_subgroups = MIN_THREADS_FOR_OCCUPANCY.div_ceil(subgroup_size);
    subgroups_per_row = cmp::max(subgroups_per_row, min_subgroups);
    subgroups_per_row = cmp::min(subgroups_per_row, max_subgroups_hw);

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

    // CRITICAL: Cap cache size at 4 to avoid register spills (user requested max=4)
    let capped_per_thread_lines = cmp::min(per_thread_lines, 4);

    let cube_dim = CubeDim::new_1d(threads_per_row);
    let cube_count = CubeCount::new_1d(num_rows_u32);

    // Detailed diagnostic logging (only once per config)
    if env::var("CUBECL_RMS_LOG").is_ok() {
        let detail_key = format!("{:?}_{}_{}_detail", input.shape, core::mem::size_of::<F>(), vectorization);
        let mut logged = LOGGED_CONFIGS.lock().unwrap();
        if !logged.contains(&detail_key) {
            eprintln!("[RMSNorm] dtype_size={} line_size={} lines_per_row={} threads_per_row={} subgroups={} per_thread_lines={} (capped={})",
                core::mem::size_of::<F>(), line_size, lines_per_row, threads_per_row, subgroups_per_row, per_thread_lines, capped_per_thread_lines);

            let bytes_per_line = line_size as usize * core::mem::size_of::<F>();
            let preferred_alignment = if bytes_per_line >= 32 { 32 } else if bytes_per_line >= 16 { 16 } else { 8 };

            eprintln!("[RMSNorm] Memory layout: vec={} ({} bytes/line), preferred_alignment={}B for optimal perf",
                vectorization, bytes_per_line, preferred_alignment);

            eprintln!("[RMSNorm] Tensor strides: input={:?} weight={:?} output={:?}",
                input.strides, weight.strides, output.strides);

            logged.insert(detail_key);
        }
    }

    let num_rows_arg = ScalarArg::new(num_rows_u32);
    let lines_per_row_arg = ScalarArg::new(lines_per_row);
    let axis_size_arg = ScalarArg::new(axis_size_u32);
    let eps_arg = ScalarArg::new(epsilon);
    let allow_native_rsqrt =
        props.features.plane.contains(Plane::Ops) && props.hardware.plane_size_min > 1;
    // Enable fast rsqrt for all dtypes (we accumulate in f32, so safe for f16/bf16 too)
    let use_fast_rsqrt = allow_native_rsqrt;
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

            // Choose kernel variant based on problem size and env override
            let use_smem = if let Ok(variant) = env::var("CUBECL_RMS_VARIANT") {
                match variant.as_str() {
                    "stream" => false,
                    "smem" => true,
                    _ => should_use_smem_variant(lines_per_row, true, line_size),
                }
            } else {
                should_use_smem_variant(lines_per_row, true, line_size)
            };

            if use_smem {
                // Variant logging removed - V2 is currently disabled anyway
                rms_norm_bias_kernel_smem::launch_unchecked::<F, R>(
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
                // V1 streaming (default)
                rms_norm_bias_kernel_stream::launch_unchecked::<F, R>(
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
            }
        } else {
            // Choose kernel variant based on problem size and env override
            let use_smem = if let Ok(variant) = env::var("CUBECL_RMS_VARIANT") {
                match variant.as_str() {
                    "stream" => false,
                    "smem" => true,
                    _ => should_use_smem_variant(lines_per_row, false, line_size),
                }
            } else {
                should_use_smem_variant(lines_per_row, false, line_size)
            };

            if use_smem {
                // V2 smem (disabled by heuristic)
                rms_norm_kernel_smem::launch_unchecked::<F, R>(
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
            } else {
                // V1 streaming (default)
                rms_norm_kernel_stream::launch_unchecked::<F, R>(
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
}
