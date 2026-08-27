use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::tensor::ErasedTensor;

/// Write through an [`ErasedTensor`] built from a launched tensor.
///
/// The degenerate case — a destination that *is* memory, reached through the
/// indirection anyway — and the one the interesting cases are compared against.
#[cube(launch_unchecked)]
fn kernel_write_of_tensor<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    out: &mut Tensor<Vector<F, N>>,
) {
    if ABSOLUTE_POS < input.len() {
        let mut sink = ErasedTensor::<F, ReadWrite>::of_tensor_mut::<N>(out);
        sink.write::<N>(ABSOLUTE_POS, input[ABSOLUTE_POS]);
    }
}

/// Read through an [`ErasedTensor`], which is the half the `IO` marker gates.
#[cube(launch_unchecked)]
fn kernel_read_of_tensor<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    out: &mut Tensor<Vector<F, N>>,
) {
    if ABSOLUTE_POS < input.len() {
        let source = ErasedTensor::<F, ReadOnly>::of_tensor::<N>(input);
        out[ABSOLUTE_POS] = source.read::<N>(ABSOLUTE_POS);
    }
}

/// Read and write through erased tensors in one kernel, with neither width
/// named in a binding.
#[cube(launch_unchecked)]
fn kernel_copy_erased<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    out: &mut Tensor<Vector<F, N>>,
) {
    let source = ErasedTensor::<F, ReadOnly>::of_tensor::<N>(input);
    let mut sink = ErasedTensor::<F, ReadWrite>::of_tensor_mut::<N>(out);

    if ABSOLUTE_POS < source.len() {
        sink.write::<N>(ABSOLUTE_POS, source.read::<N>(ABSOLUTE_POS));
    }
}

/// The `len` an erased tensor reports is in lines, so a bound check against the
/// caller's line index is allowed to use it. Getting this wrong would be off by
/// a factor of the width.
#[cube(launch_unchecked)]
fn kernel_len<F: Float, N: Size>(input: &Tensor<Vector<F, N>>, out: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        let source = ErasedTensor::<F, ReadOnly>::of_tensor::<N>(input);
        out[0] = u32::cast_from(source.len());
    }
}

const LINES: usize = 8;

fn values(width: usize) -> Vec<f32> {
    (0..LINES * width).map(|i| i as f32 + 0.5).collect()
}

/// Each kernel has to land the same bytes a direct copy would, so a wrong width
/// or a store on the wrong element shows up as a mismatch rather than as a
/// plausible-looking buffer.
pub fn test_write_of_tensor<R: Runtime>(client: ComputeClient<R>, width: usize) {
    let input = values(width);
    let actual = launch_copy::<R>(&client, &input, width, Kernel::Write);
    assert_eq!(
        actual, input,
        "write through of_tensor_mut at width {width}"
    );
}

pub fn test_read_of_tensor<R: Runtime>(client: ComputeClient<R>, width: usize) {
    let input = values(width);
    let actual = launch_copy::<R>(&client, &input, width, Kernel::Read);
    assert_eq!(actual, input, "read through of_tensor at width {width}");
}

pub fn test_copy_erased<R: Runtime>(client: ComputeClient<R>, width: usize) {
    let input = values(width);
    let actual = launch_copy::<R>(&client, &input, width, Kernel::Copy);
    assert_eq!(actual, input, "read and write erased at width {width}");
}

pub fn test_len_is_in_lines<R: Runtime>(client: ComputeClient<R>, width: usize) {
    let input = values(width);

    let handle_in = client.create_from_slice(f32::as_bytes(&input));
    let handle_out = client.empty(size_of::<u32>());

    unsafe {
        kernel_len::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            width,
            TensorArg::from_raw_parts(handle_in, vec![1].into(), vec![input.len()].into()),
            BufferArg::from_raw_parts(handle_out.clone(), 1),
        )
    };

    let out = client.read_one_unchecked(handle_out);
    // A kernel that failed to compile still reads back "ok", so pin the length
    // before trusting the value.
    assert_eq!(out.len(), size_of::<u32>(), "kernel produced no output");
    assert_eq!(
        u32::from_bytes(&out)[0] as usize,
        LINES,
        "len at width {width} should be in lines, not scalars"
    );
}

enum Kernel {
    Write,
    Read,
    Copy,
}

fn launch_copy<R: Runtime>(
    client: &ComputeClient<R>,
    input: &[f32],
    width: usize,
    kernel: Kernel,
) -> Vec<f32> {
    let lines = input.len() / width;
    let handle_in = client.create_from_slice(f32::as_bytes(input));
    let handle_out = client.empty(size_of_val(input));

    let cube_dim = 32u32;
    let cube_count = CubeCount::Static((lines as u32).div_ceil(cube_dim), 1, 1);

    unsafe {
        let shape = vec![input.len()];
        let arg_in = TensorArg::from_raw_parts(handle_in, vec![1].into(), shape.clone().into());
        let arg_out = TensorArg::from_raw_parts(handle_out.clone(), vec![1].into(), shape.into());
        let dim = CubeDim::new_1d(cube_dim);

        match kernel {
            Kernel::Write => kernel_write_of_tensor::launch_unchecked::<f32, R>(
                client, cube_count, dim, width, arg_in, arg_out,
            ),
            Kernel::Read => kernel_read_of_tensor::launch_unchecked::<f32, R>(
                client, cube_count, dim, width, arg_in, arg_out,
            ),
            Kernel::Copy => kernel_copy_erased::launch_unchecked::<f32, R>(
                client, cube_count, dim, width, arg_in, arg_out,
            ),
        }
    }

    let out = client.read_one_unchecked(handle_out);
    // A kernel that failed to compile still reads back "ok"; the length is what
    // catches it.
    assert_eq!(
        out.len(),
        size_of_val(input),
        "kernel produced no output, so it never ran"
    );
    f32::from_bytes(&out).to_vec()
}

#[macro_export]
macro_rules! testgen_erased {
    () => {
        $crate::testgen_erased!(width_1 => 1, width_4 => 4);
    };
    ($($name:ident => $width:expr),*) => {
        mod erased {
            use super::*;

            $(
                mod $name {
                    use super::*;

                    #[$crate::tests::test_log::test]
                    fn write_of_tensor() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_std::tests::erased::test_write_of_tensor::<TestRuntime>(
                            client, $width,
                        );
                    }

                    #[$crate::tests::test_log::test]
                    fn read_of_tensor() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_std::tests::erased::test_read_of_tensor::<TestRuntime>(
                            client, $width,
                        );
                    }

                    #[$crate::tests::test_log::test]
                    fn copy_erased() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_std::tests::erased::test_copy_erased::<TestRuntime>(client, $width);
                    }

                    #[$crate::tests::test_log::test]
                    fn len_is_in_lines() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_std::tests::erased::test_len_is_in_lines::<TestRuntime>(
                            client, $width,
                        );
                    }
                }
            )*
        }
    };
}
