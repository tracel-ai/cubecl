use std::fs;

use cubecl_core::{self as cubecl, Compiler, ExecutionMode};
use cubecl_core::{cube, prelude::*};

mod common;
use common::*;
use cubecl_wgpu::WgpuRuntime;
use rspirv::binary::Disassemble;

#[cube(create_dummy_kernel, launch_unchecked)]
pub fn assign_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if UNIT_POS == 0 {
        output[0] = input[0];
    }
}

#[test]
pub fn assign() {
    let client = client();
    let input = handle(&client);
    let output = handle(&client);

    let kernel = assign_kernel::create_dummy_kernel::<WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        tensor(&input),
        tensor(&output),
    );

    let kernel = compile(kernel);
    fs::write("out/test.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/test.spv", to_bytes(kernel.clone())).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn slice_len_kernel(input: &Array<f32>, output: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let slice = input.slice(2, 4);
        let _tmp = slice[0]; // It must be used at least once, otherwise wgpu isn't happy.
        output[0] = slice.len();
    }
}

#[test]
pub fn slice_len() {
    let client = client();
    let input = handle(&client);
    let output = handle(&client);

    let kernel = slice_len_kernel::create_dummy_kernel::<WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        array(&input),
        array(&output),
    );
    let kernel = compile(kernel);
    fs::write("out/slice_len.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/slice_len.spv", to_bytes(kernel.clone())).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn slice_for_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        let mut sum = 0f32;

        for item in input.slice(2, 4) {
            sum += item;
        }

        output[0] = sum;
    }
}

#[test]
pub fn slice_for() {
    let client = client();
    let input = handle(&client);
    let output = handle(&client);

    let kernel = slice_for_kernel::create_dummy_kernel::<WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        array(&input),
        array(&output),
    );
    let kernel = compile_unchecked(kernel);
    fs::write("out/slice_for.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/slice_for.spv", to_bytes(kernel.clone())).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}

#[cube(launch_unchecked, create_dummy_kernel)]
fn test_dot_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < rhs.len() {
        output[ABSOLUTE_POS] = F::dot(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

#[test]
pub fn test_dot() {
    let client = client();
    let lhs = handle(&client);
    let rhs = handle(&client);
    let output = handle(&client);

    let kernel = test_dot_kernel::create_dummy_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        array_vec(&lhs, 2),
        array_vec(&rhs, 2),
        array(&output),
    );
    let wgsl = <<WgpuRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Unchecked,
    )
    .to_string();
    let kernel = compile_unchecked(kernel);
    fs::write("out/dot.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/dot.spv", to_bytes(kernel.clone())).unwrap();
    fs::write("out/dot.wgsl", wgsl).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}

#[cube(launch_unchecked, create_dummy_kernel)]
fn test_unary<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = F::normalize(input[ABSOLUTE_POS]);
    }
}

#[test]
pub fn unary() {
    let client = client();
    let input = handle(&client);
    let output = handle(&client);

    let kernel = test_unary::create_dummy_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        array(&input),
        array(&output),
    );

    let wgsl = <<WgpuRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Unchecked,
    )
    .to_string();
    let kernel = compile_unchecked(kernel);
    fs::write("out/normalize.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/normalize.spv", to_bytes(kernel.clone())).unwrap();
    fs::write("out/normalize.wgsl", wgsl).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}

#[cube(launch_unchecked, create_dummy_kernel)]
pub fn kernel_absolute_pos(output1: &mut Array<u32>) {
    output1[ABSOLUTE_POS] = ABSOLUTE_POS;
}

#[test]
pub fn absolute_pos() {
    let client = client();
    let output = handle(&client);

    let kernel = kernel_absolute_pos::create_dummy_kernel::<WgpuRuntime>(
        CubeCount::Static(3, 5, 2),
        CubeDim::new(16, 16, 1),
        array(&output),
    );

    let wgsl = <<WgpuRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Unchecked,
    )
    .to_string();
    let kernel = compile_unchecked(kernel);
    fs::write("out/absolute_pos.spv.txt", kernel.clone().disassemble()).unwrap();
    fs::write("out/absolute_pos.spv", to_bytes(kernel.clone())).unwrap();
    fs::write("out/absolute_pos.wgsl", wgsl).unwrap();

    let expected = include_str!("slice_assign.spv.text").replace("\r\n", "\n");
    assert_eq!(kernel.disassemble(), expected);
}
