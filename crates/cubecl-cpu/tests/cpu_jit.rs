//! Link and execute the CPU compiler with only the host LLVM target available.
use cubecl_core::{self as cubecl, prelude::*};
use cubecl_cpu::CpuRuntime;
use cubecl_environment::future::block_on;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn affine(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0 + 1.0;
    }
}

#[test]
fn cpu_jit_executes_across_cubes_and_handles_a_partial_cube() {
    let client = CpuRuntime::client(&Default::default());
    let input: Vec<f32> = (0..259).map(|i| i as f32 - 128.0).collect();
    let source = client.create_from_slice(f32::as_bytes(&input));
    let output = client.empty(input.len() * size_of::<f32>());

    let cube_dim = CubeDim::new(&client, 4);
    let cubes = input.len().div_ceil(cube_dim.num_elems() as usize);
    affine::launch(
        &client,
        CubeCount::new_1d(cubes as u32),
        cube_dim,
        // SAFETY: both buffers contain input.len() f32 slots. The kernel bounds
        // checks any units in the final cube that are outside those slots.
        unsafe { BufferArg::from_raw_parts(source, input.len()) },
        unsafe { BufferArg::from_raw_parts(output.clone(), input.len()) },
    );
    block_on(client.sync_buffers([&output])).unwrap();
    let bytes = client.read_one(output).unwrap();
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0 + 1.0).collect();
    assert_eq!(f32::from_bytes(&bytes), expected);
}
