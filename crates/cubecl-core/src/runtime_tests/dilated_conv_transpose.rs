//! Dilated conv-transpose inner loops used by burn-cubecl's conv1d dgrad fallback.
//!
//! `conv_transpose2d_direct` enumerates the *dilated* input window and then
//! keeps only the taps where `numerator.is_multiple_of(dilation)`. Trip count
//! is therefore `kernel * dilation`, of which only `kernel` iterations do a
//! multiply-add.
//!
//! `QuotientRangePass` re-indexes that loop by the quotient so it runs once per
//! tap. These tests pin the semantics it has to preserve; the trip count itself
//! is checked on the IR in `cubecl-opt/tests/quotient_range.rs`.
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
#[derive(Clone, Copy)]
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

/// Writes `[total, useful]` iteration counts of the filter loop at `out_y`.
#[cube(launch)]
fn conv_transpose1d_filter_trip_count_kernel(output: &mut [u32], out_y: u32, args: ConvArgs) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let out_y = out_y as usize;
    let stride = args.stride as usize;
    let dilation = args.dilation as usize;
    let padding = args.padding as usize;
    let in_len = args.in_len as usize;
    let kernel = args.kernel as usize;

    let stride_i = args.stride as i32;
    let kms = (kernel * dilation) as i32 - stride_i;
    let y_start = ((out_y + padding) as i32 - kms) / stride_i;
    let y_end = clamp(kms + y_start + 1, 0, in_len as i32) as usize;
    let y_start = clamp_min(y_start, 0) as usize;

    let numerator_base = out_y + padding;
    let mut total = 0u32;
    let mut useful = 0u32;

    for in_y in y_start..y_end {
        total += 1;
        let numerator_tmp = in_y * stride;
        if numerator_base >= numerator_tmp {
            let numerator = numerator_base - numerator_tmp;
            if numerator.is_multiple_of(dilation) {
                let kernel_y = numerator / dilation;
                if kernel_y < kernel {
                    useful += 1;
                }
            }
        }
    }

    output[0] = total;
    output[1] = useful;
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

fn filter_trip_count_ref(problem: Problem, out_y: u32) -> (u32, u32) {
    let stride = problem.stride as i32;
    let dilation = problem.dilation as i32;
    let padding = problem.padding as i32;
    let in_len = problem.in_len as i32;
    let kernel = problem.kernel as i32;
    let out_y = out_y as i32;

    let kms = kernel * dilation - stride;
    let y_start = (out_y + padding - kms) / stride;
    let y_end = (kms + y_start + 1).clamp(0, in_len);
    let y_start = y_start.max(0);

    let numerator_base = out_y + padding;
    let mut total = 0u32;
    let mut useful = 0u32;
    for in_y in y_start..y_end {
        total += 1;
        let numerator_tmp = in_y * stride;
        if numerator_base >= numerator_tmp {
            let numerator = numerator_base - numerator_tmp;
            if numerator % dilation == 0 {
                let kernel_y = numerator / dilation;
                if kernel_y >= 0 && kernel_y < kernel {
                    useful += 1;
                }
            }
        }
    }
    (total, useful)
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

fn run_and_read<K>(client: &Client, kernel: K, problem: Problem) -> Vec<f32>
where
    K: Fn(&Client, CubeCount, CubeDim, BufferArg, BufferArg, BufferArg, ConvArgsLaunch),
{
    let input_data = fill_input(problem);
    let weight_data = fill_weight(problem);
    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let weight = client.create_from_slice(f32::as_bytes(&weight_data));
    let output = client.empty(problem.out_len() as usize * core::mem::size_of::<f32>());
    let output = launch_kernel(client, kernel, &input, &weight, output, problem);
    let bytes = client.read_one_unchecked(output);
    f32::from_bytes(&bytes).to_vec()
}

fn assert_matches_reference<K>(client: &Client, kernel: K, problem: Problem, label: &str)
where
    K: Fn(&Client, CubeCount, CubeDim, BufferArg, BufferArg, BufferArg, ConvArgsLaunch),
{
    let input = fill_input(problem);
    let weight = fill_weight(problem);
    let expected = reference(problem, &input, &weight);
    let actual = run_and_read(client, kernel, problem);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch for {problem:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = 1e-4f32 * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: actual={a} expected={e} problem={problem:?}"
        );
    }
}

impl core::fmt::Debug for Problem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "in_c={} in_len={} k={} d={} pad={} stride={} out={}",
            self.in_c,
            self.in_len,
            self.kernel,
            self.dilation,
            self.padding,
            self.stride,
            self.out_len()
        )
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

/// Both loop styles must match the kernel-position CPU reference, including
/// non-power-of-two dilation (runtime `is_multiple_of`).
pub fn test_filter_loop_matches_reference<R: Runtime>(client: Client) {
    for problem in correctness_cases() {
        assert_matches_reference(
            &client,
            conv_transpose1d_filter_kernel::launch,
            problem,
            "filter loop",
        );
    }
}

/// Interior output of a same-size `k=5` problem: the filter loop runs `5d`
/// times and keeps 5. That is the wasted-iteration claim, as a count rather
/// than a wall-clock measurement.
pub fn test_filter_trip_count_scales_with_dilation<R: Runtime>(client: Client) {
    let kernel = 5u32;
    // Window width is `kernel * dilation`; length must keep an interior
    // `out_y` unclipped at dilation 8 (`5 * 8 = 40`).
    let length = 64u32;
    let out_y = length / 2;

    for dilation in [1u32, 2, 3, 4, 8] {
        let problem = Problem::same_size(1, length, kernel, dilation);
        let handle = client.empty(2 * core::mem::size_of::<u32>());
        conv_transpose1d_filter_trip_count_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
            out_y,
            args_launch(problem),
        );
        let actual = client.read_one_unchecked(handle);
        let actual = u32::from_bytes(&actual);
        let (total, useful) = filter_trip_count_ref(problem, out_y);

        assert_eq!(
            actual,
            &[total, useful],
            "trip count at dilation={dilation}: gpu=[{}, {}] ref=[{total}, {useful}]",
            actual[0],
            actual[1]
        );
        assert_eq!(
            useful, kernel,
            "interior out_y={out_y} should keep every kernel tap, dilation={dilation}"
        );
        assert_eq!(
            total,
            kernel * dilation,
            "filter window at dilation={dilation} should be kernel*dilation, got {total}"
        );
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
            fn test_filter_loop_matches_reference() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::dilated_conv_transpose::test_filter_loop_matches_reference::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_filter_trip_count_scales_with_dilation() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::dilated_conv_transpose::test_filter_trip_count_scales_with_dilation::<
                    TestRuntime,
                >(client);
            }
        }
    };
}
