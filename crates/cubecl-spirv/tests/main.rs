use std::fs;

use cubecl_core::{prelude::ScalarArg, CubeCount, CubeDim};
use cubecl_linalg::matmul::tests::{
    make_tiling2d_config, tiling2d::load_shared_memory::load_tensor_test,
};
use cubecl_wgpu::WgpuRuntime;

mod common;
use common::*;
use rspirv::binary::Disassemble;

#[test]
pub fn load_lhs_plain() {
    let client = client();
    let lhs = handle(&client);
    let rhs = handle(&client);

    let config = make_tiling2d_config(6, 14, 8);

    let kernel = load_tensor_test::create_dummy_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        tensor_vec(&lhs, 4),
        array_vec(&rhs, 4),
        ScalarArg::new(1),
        ScalarArg::new(4),
        ScalarArg::new(8),
        config,
        true,
    );

    let kernel = compile_unchecked(kernel);
    fs::write("out/lhs_plain.spv", as_bytes(&kernel)).unwrap();
    fs::write("out/lhs_plain.txt", kernel.disassemble()).unwrap();

    panic!()
    //let expected = include_str!("unary_bench.wgsl").replace("\r\n", "\n");
    //assert_eq!(compile(kernel), expected);
}
