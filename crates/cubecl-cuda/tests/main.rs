use std::num::NonZero;

use common::*;
use constant_array_kernel::ConstantArrayKernel;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cuda::CudaRuntime;
use execute_unary_kernel::ExecuteUnaryKernel;
use half::bf16;
use half::f16;
use kernel_sum::KernelSum;
use naming_kernel::NamingKernel;
use pretty_assertions::assert_eq;
use sequence_for_loop_kernel::SequenceForLoopKernel;
use slice_assign_kernel::SliceAssignKernel;

mod common;

#[cube(launch_unchecked, create_dummy_kernel)]
pub fn slice_assign_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if UNIT_POS == 0 {
        let mut slice_1 = output.slice_mut(2, 3);
        slice_1[0] = input[0];
    }
}

#[test]
pub fn slice_assign() {
    let kernel = SliceAssignKernel::<CudaRuntime>::new(settings(), tensor(), tensor());
    let expected = include_str!("slice_assign.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn kernel_sum(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = cubecl_core::prelude::plane_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[test]
pub fn plane_sum() {
    let kernel = KernelSum::<CudaRuntime>::new(settings(), tensor());

    let expected = include_str!("plane_sum.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn sequence_for_loop_kernel(output: &mut Array<f32>) {
    if UNIT_POS != 0 {
        return;
    }

    let mut sequence = Sequence::<f32>::new();
    sequence.push(1.0);
    sequence.push(4.0);

    for value in sequence {
        output[0] += value;
    }
}

#[test]
pub fn sequence_for_loop() {
    let kernel = SequenceForLoopKernel::<CudaRuntime>::new(settings(), array());
    let expected = include_str!("sequence_for_loop.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn execute_unary_kernel<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in 0..256u32 {
            if i % 2 == 0 {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

#[test]
pub fn unary_bench() {
    let kernel = ExecuteUnaryKernel::<f32, CudaRuntime>::new(
        settings(),
        tensor_vec(4),
        tensor_vec(4),
        tensor_vec(4),
    );
    let expected = include_str!("unary_bench.cu").replace("\r\n", "\n");
    let expected = expected.trim();

    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn constant_array_kernel<F: Float>(out: &mut Tensor<F>, #[comptime] data: Vec<u32>) {
    let array = Array::<F>::from_data(data);

    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = array[ABSOLUTE_POS];
    }
}

#[test]
pub fn constant_array() {
    let data: Vec<u32> = vec![3, 5, 1];

    let kernel = ConstantArrayKernel::<f32, CudaRuntime>::new(settings(), tensor(), data);
    let expected = include_str!("constant_array.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

// This kernel just exists to have a few generics in order to observe
// that the generics get propagated into the WGSL kernel name
#[allow(clippy::extra_unused_type_parameters)]
#[cube(launch, create_dummy_kernel)]
fn naming_kernel<F1: Float, N1: Numeric, F2: Float, N2: Numeric>(out: &mut Array<F1>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F1::from_int(0);
    }
}

#[test]
pub fn naming() {
    let kernel = NamingKernel::<f32, u8, bf16, i64, CudaRuntime>::new(settings(), array());
    let expected = include_str!("naming.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn clamp_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    out[ABSOLUTE_POS] = F::clamp(input[0], F::from_int(0), F::from_int(2));
}

#[test]
pub fn test_clamp() {
    let kernel = clamp_kernel::ClampKernel::<f32, CudaRuntime>::new(settings(), array(), array());
    let expected = include_str!("clamp_f32.cu").replace("\r\n", "\n");
    assert_eq!(compile(kernel), expected);

    let kernel = clamp_kernel::ClampKernel::<f16, CudaRuntime>::new(settings(), array(), array());
    let expected = include_str!("clamp_f16.cu").replace("\r\n", "\n");
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn lined_clamp_kernel<F: Float>(input: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    out[ABSOLUTE_POS] = Line::<F>::clamp(
        input[0],
        Line::new(F::from_int(0)),
        Line::new(F::from_int(2)),
    );
}

#[test]
pub fn test_lined_clamp() {
    let arg4 = ArrayCompilationArg {
        inplace: None,
        vectorisation: NonZero::new(4),
    };

    let kernel = lined_clamp_kernel::LinedClampKernel::<f32, CudaRuntime>::new(
        settings(),
        arg4.clone(),
        arg4.clone(),
    );

    let expected = include_str!("lined_clamp_f32.cu").replace("\r\n", "\n");
    assert_eq!(compile(kernel), expected);

    let kernel = lined_clamp_kernel::LinedClampKernel::<f16, CudaRuntime>::new(
        settings(),
        arg4.clone(),
        arg4.clone(),
    );
    // TODO: Regenerate when correct
    // std::fs::write("tests/lined_clamp_f16.cu", compile(kernel));

    let expected = include_str!("lined_clamp_f16.cu").replace("\r\n", "\n");
    assert_eq!(compile(kernel), expected);
}
