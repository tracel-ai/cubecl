//! Dilated conv-transpose inner loops used by burn-cubecl's conv1d dgrad fallback.
//!
//! `conv_transpose2d_direct` enumerates the *dilated* input window and then
//! keeps only the taps where `numerator.is_multiple_of(dilation)`. Trip count
//! is therefore `kernel * dilation`, of which only `kernel` iterations do a
//! multiply-add. The other loop enumerates kernel positions and back-computes
//! the input index, so its trip count is just `kernel`.
//!
//! Same-size problems (odd kernel, `padding = dilation * (kernel - 1) / 2`,
//! stride 1) keep tensor shapes and MAC count fixed while dilation changes.

use crate::{self as cubecl};
use alloc::vec;
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;

/// 1-D conv-transpose hyperparameters. Layouts are `[in_c, in_len]` and
/// `[in_c, kernel]`, one output channel.
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub in_c: u32,
    pub in_len: u32,
    pub kernel: u32,
    pub dilation: u32,
    pub padding: u32,
    pub stride: u32,
}

impl Problem {
    /// Same-length transpose: `out_len == in_len`. `kernel` must be odd.
    pub fn same_size(in_c: u32, length: u32, kernel: u32, dilation: u32) -> Self {
        assert!(kernel % 2 == 1, "same-size padding needs an odd kernel");
        Self {
            in_c,
            in_len: length,
            kernel,
            dilation,
            padding: dilation * (kernel - 1) / 2,
            stride: 1,
        }
    }

    pub fn out_len(&self) -> u32 {
        (self.in_len - 1) * self.stride + self.dilation * (self.kernel - 1) - 2 * self.padding + 1
    }
}

#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    stride: u32,
    dilation: u32,
    padding: u32,
    in_c: u32,
    in_len: u32,
    kernel: u32,
}

/// Input-window loop from `conv_transpose2d_direct_kernel`: `y_end - y_start`
/// is `kernel * dilation`, then `is_multiple_of(dilation)` drops the holes.
#[cube(launch)]
fn conv_transpose1d_filter_kernel(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    args: ConvArgs,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_y = ABSOLUTE_POS;
    let stride = args.stride as usize;
    let dilation = args.dilation as usize;
    let padding = args.padding as usize;
    let in_len = args.in_len as usize;
    let kernel = args.kernel as usize;
    let in_c = args.in_c as usize;

    let stride_i = args.stride as i32;
    let kms = (kernel * dilation) as i32 - stride_i;
    let y_start = ((out_y + padding) as i32 - kms) / stride_i;
    let y_end = clamp(kms + y_start + 1, 0, in_len as i32) as usize;
    let y_start = clamp_min(y_start, 0) as usize;

    let numerator_base = out_y + padding;
    let mut sum = 0.0f32;

    for ic in 0..in_c {
        let in_row = ic * in_len;
        let w_row = ic * kernel;
        for in_y in y_start..y_end {
            let numerator_tmp = in_y * stride;
            if numerator_base >= numerator_tmp {
                let numerator = numerator_base - numerator_tmp;
                if numerator.is_multiple_of(dilation) {
                    let kernel_y = numerator / dilation;
                    if kernel_y < kernel {
                        sum += input[in_row + in_y] * weight[w_row + kernel_y];
                    }
                }
            }
        }
    }

    output[out_y] = sum;
}

/// Kernel-position loop: trip count is `kernel`, independent of dilation.
#[cube(launch)]
fn conv_transpose1d_kernel_pos_kernel(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    args: ConvArgs,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_y = ABSOLUTE_POS;
    let stride = args.stride as usize;
    let dilation = args.dilation as usize;
    let padding = args.padding as usize;
    let in_len = args.in_len as usize;
    let kernel = args.kernel as usize;
    let in_c = args.in_c as usize;

    let numerator_base = (out_y + padding) as i32;
    let mut sum = 0.0f32;

    for ic in 0..in_c {
        let in_row = ic * in_len;
        let w_row = ic * kernel;
        for k in 0..kernel {
            let in_pos = numerator_base - (k * dilation) as i32;
            if in_pos >= 0 && (in_pos as usize).is_multiple_of(stride) {
                let in_y = in_pos as usize / stride;
                if in_y < in_len {
                    sum += input[in_row + in_y] * weight[w_row + k];
                }
            }
        }
    }

    output[out_y] = sum;
}

fn args_launch(problem: Problem) -> ConvArgsLaunch {
    ConvArgsLaunch::new(
        problem.stride,
        problem.dilation,
        problem.padding,
        problem.in_c,
        problem.in_len,
        problem.kernel,
    )
}

fn launch_config(n: u32) -> (CubeCount, CubeDim) {
    let cube_dim = 256u32.min(n.max(1));
    (
        CubeCount::Static(n.div_ceil(cube_dim), 1, 1),
        CubeDim::new_1d(cube_dim),
    )
}

fn fill_input(problem: Problem) -> Vec<f32> {
    (0..problem.in_c * problem.in_len)
        .map(|i| (i % 7) as f32 + 1.0)
        .collect()
}

fn fill_weight(problem: Problem) -> Vec<f32> {
    (0..problem.in_c * problem.kernel)
        .map(|i| (i % 5) as f32 + 0.5)
        .collect()
}

fn reference(problem: Problem, input: &[f32], weight: &[f32]) -> Vec<f32> {
    let out_len = problem.out_len() as usize;
    let in_c = problem.in_c as usize;
    let in_len = problem.in_len as usize;
    let kernel = problem.kernel as usize;
    let dilation = problem.dilation as usize;
    let padding = problem.padding as usize;
    let stride = problem.stride as usize;
    let mut output = vec![0.0f32; out_len];

    for out_y in 0..out_len {
        let mut sum = 0.0f32;
        let numerator_base = (out_y + padding) as i32;
        for ic in 0..in_c {
            let in_row = ic * in_len;
            let w_row = ic * kernel;
            for k in 0..kernel {
                let in_pos = numerator_base - (k * dilation) as i32;
                if in_pos < 0 {
                    continue;
                }
                if (in_pos as usize) % stride != 0 {
                    continue;
                }
                let in_y = in_pos as usize / stride;
                if in_y < in_len {
                    sum += input[in_row + in_y] * weight[w_row + k];
                }
            }
        }
        output[out_y] = sum;
    }
    output
}

fn launch_kernel<K>(
    client: &Client,
    kernel: K,
    input: &Handle,
    weight: &Handle,
    output: Handle,
    problem: Problem,
) -> Handle
where
    K: Fn(&Client, CubeCount, CubeDim, BufferArg, BufferArg, BufferArg, ConvArgsLaunch),
{
    let out_len = problem.out_len();
    let (count, dim) = launch_config(out_len);
    kernel(
        client,
        count,
        dim,
        unsafe {
            BufferArg::from_raw_parts(input.clone(), (problem.in_c * problem.in_len) as usize)
        },
        unsafe {
            BufferArg::from_raw_parts(weight.clone(), (problem.in_c * problem.kernel) as usize)
        },
        unsafe { BufferArg::from_raw_parts(output.clone(), out_len as usize) },
        args_launch(problem),
    );
    output
}

/// Allocate input/weight/output for `problem`. Values are deterministic so a
/// timed launch still writes a real reduction the compiler cannot drop.
pub fn prepare_buffers(client: &Client, problem: Problem) -> (Handle, Handle, Handle) {
    let input = client.create_from_slice(f32::as_bytes(&fill_input(problem)));
    let weight = client.create_from_slice(f32::as_bytes(&fill_weight(problem)));
    let output = client.empty(problem.out_len() as usize * core::mem::size_of::<f32>());
    (input, weight, output)
}

/// Launch the input-window filter kernel for correctness checks and benchmarks.
pub fn launch_filter(
    client: &Client,
    input: &Handle,
    weight: &Handle,
    output: Handle,
    problem: Problem,
) -> Handle {
    launch_kernel(
        client,
        conv_transpose1d_filter_kernel::launch,
        input,
        weight,
        output,
        problem,
    )
}

/// Launch the kernel-position reference loop for benchmarks.
pub fn launch_kernel_pos(
    client: &Client,
    input: &Handle,
    weight: &Handle,
    output: Handle,
    problem: Problem,
) -> Handle {
    launch_kernel(
        client,
        conv_transpose1d_kernel_pos_kernel::launch,
        input,
        weight,
        output,
        problem,
    )
}

fn assert_matches_reference(client: &Client, problem: Problem) {
    let input_data = fill_input(problem);
    let weight_data = fill_weight(problem);
    let expected = reference(problem, &input_data, &weight_data);
    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let weight = client.create_from_slice(f32::as_bytes(&weight_data));
    let output = client.empty(expected.len() * core::mem::size_of::<f32>());
    let output = launch_filter(client, &input, &weight, output, problem);
    let bytes = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&bytes);
    assert_eq!(actual.len(), expected.len(), "{problem:?}");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let tol = 1e-4f32 * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tol,
            "index {i}: actual={a} expected={e} problem={problem:?}"
        );
    }
}

fn correctness_cases() -> [Problem; 6] {
    [
        Problem::same_size(3, 16, 5, 1),
        Problem::same_size(3, 16, 5, 2),
        Problem::same_size(3, 16, 5, 3),
        Problem::same_size(3, 16, 5, 4),
        Problem::same_size(4, 20, 5, 8),
        Problem {
            in_c: 2,
            in_len: 12,
            kernel: 3,
            dilation: 2,
            padding: 1,
            stride: 2,
        },
    ]
}

/// A rolling checksum checks the sequence of surviving taps, including interior ones.
/// All bounds are runtime arguments so the compiler must emit its safety checks.
#[cube(launch)]
fn guarded_range_kernel(
    output: &mut [u32],
    lo: u32,
    hi: u32,
    base: u32,
    stride: u32,
    divisor: u32,
    #[comptime] stop_after_first: bool,
    #[comptime] count_source: bool,
) {
    let base = base as usize;
    let stride = stride as usize;
    let divisor = divisor as usize;
    for i in lo as usize..hi as usize {
        if count_source {
            // This effect forces the pass to leave the reference loop alone.
            output[1] += 1;
        }
        let scaled = i * stride;
        if base >= scaled {
            let numerator = base - scaled;
            if numerator.is_multiple_of(divisor) {
                output[0] = output[0] * 31 + i as u32;
                if stop_after_first {
                    terminate!();
                }
            }
        }
    }
}

pub fn test_quotient_range_boundaries<R: Runtime>(client: Client) {
    for (lo, hi, base, stride, divisor) in [
        (9u32, 49u32, 48u32, 1u32, 8u32),
        (1 << 31, (1 << 31) + 8, 16, 2, 8),
        ((1 << 31) - 8, 1 << 31, u32::MAX, 2, 8),
        (u32::MAX - 8, u32::MAX, u32::MAX, 1, 8),
        (0, 8, u32::MAX, 1, 1),
        // Exercise recovery with both coprime and shared stride/divisor factors.
        (0, 30, 80, 3, 8),
        (0, 16, 32, 2, 4),
        (0, 8, 48, 0, 8),
        (9, 0, u32::MAX, 2, 8),
        (0, 0, 0, 0, 0),
    ] {
        for stop_after_first in [false, true] {
            let mut expected = u32::MAX;
            // Compare on the device: backends can lower cube.index to different
            // widths, so host u32 wrapping is not a portable source reference.
            for count_source in [true, false] {
                let output = client.create_from_slice(u32::as_bytes(&[u32::MAX, 0]));
                guarded_range_kernel::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
                    lo,
                    hi,
                    base,
                    stride,
                    divisor,
                    stop_after_first,
                    count_source,
                );
                let actual = client.read_one_unchecked(output);
                let actual = u32::from_bytes(&actual)[0];
                if count_source {
                    expected = actual;
                } else {
                    assert_eq!(
                        actual, expected,
                        "lo={lo} hi={hi} base={base} stride={stride} divisor={divisor} stop={stop_after_first}"
                    );
                }
            }
        }
    }
}

/// The filter loop must match the kernel-position CPU reference, including
/// non-power-of-two dilation (runtime `is_multiple_of`).
pub fn test_filter_loop_matches_reference<R: Runtime>(client: Client) {
    for problem in correctness_cases() {
        assert_matches_reference(&client, problem);
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_dilated_conv_transpose {
    () => {
        mod dilated_conv_transpose {
            use super::*;
            use $crate::__private::Runtime;

            #[$crate::runtime_tests::test_log::test]
            fn test_quotient_range_boundaries() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::dilated_conv_transpose::test_quotient_range_boundaries::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_filter_loop_matches_reference() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::dilated_conv_transpose::test_filter_loop_matches_reference::<
                    TestRuntime,
                >(client);
            }

        }
    };
}
