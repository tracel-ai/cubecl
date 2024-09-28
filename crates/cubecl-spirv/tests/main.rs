use std::fs;

use cubecl_core as cubecl;
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
