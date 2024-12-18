use common::*;
use constant_array_kernel::ConstantArrayKernel;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_wgpu::WgpuRuntime;
use execute_unary_kernel::ExecuteUnaryKernel;
use half::bf16;
use kernel_elect::KernelElect;
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
    let kernel = SliceAssignKernel::<WgpuRuntime>::new(settings(1, 1), tensor(), tensor());
    let expected = load_kernel_string!("slice_assign.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
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
    let kernel = KernelSum::<WgpuRuntime>::new(settings(4, 1), tensor());
    let expected = load_kernel_string!("plane_sum.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn kernel_elect(output: &mut Tensor<u32>) {
    let elected = cubecl_core::prelude::plane_elect();
    output[UNIT_POS] = elected as u32;
}

#[test]
pub fn plane_elect() {
    let kernel = KernelElect::<WgpuRuntime>::new(settings(4, 1), tensor());
    let expected = load_kernel_string!("plane_elect.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
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
    let kernel = SequenceForLoopKernel::<WgpuRuntime>::new(settings(16, 16), array());
    let expected = load_kernel_string!("sequence_for_loop.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
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
    let kernel = ExecuteUnaryKernel::<f32, WgpuRuntime>::new(
        settings(16, 16),
        tensor_vec(4),
        tensor_vec(4),
        tensor_vec(4),
    );
    let expected = load_kernel_string!("unary_bench.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
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

    let kernel = ConstantArrayKernel::<f32, WgpuRuntime>::new(settings(16, 16), tensor(), data);
    let expected = load_kernel_string!("constant_array.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
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
    let kernel = NamingKernel::<f32, u8, bf16, i64, WgpuRuntime>::new(settings(16, 16), array());
    let expected = load_kernel_string!("naming.wgsl");
    let compiled = compile(kernel);
    println!("{}", compiled);
    assert_eq!(compiled, expected);
}
