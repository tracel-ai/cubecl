//! Two kernels with nested `#[cube]` calls, for profilers and debuggers.
//!
//! `nested_lines` has the level of the cargo profile (line tables in `dev` and in a
//! profile with `debug = "line-tables-only"`). `nested_full` has `debug_symbols`, so it
//! also has the source text.

use cubecl::{Device, prelude::*};

#[cube]
fn square_third(x: f32) -> f32 {
    let y = x * x;
    y / 3.0
}

#[cube]
fn doubled(x: f32) -> f32 {
    square_third(x) * 2.0
}

#[cube(launch)]
fn nested_lines(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = doubled(input[ABSOLUTE_POS]);
    }
}

#[cube(launch, debug_symbols)]
fn nested_full(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = doubled(input[ABSOLUTE_POS]) + 1.0;
    }
}

const CUBE_DIM: u32 = 256;
const CUBE_COUNT: u32 = 4096;
const LEN: usize = (CUBE_DIM * CUBE_COUNT) as usize;

fn main() {
    let client = Device::default().client();
    let input: Vec<f32> = (0..7u8).cycle().take(LEN).map(f32::from).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(LEN * size_of::<f32>());
    let cube_count = CubeCount::Static(CUBE_COUNT, 1, 1);
    // SAFETY: each handle holds `LEN` values of `f32`.
    let buffer = |handle| unsafe { BufferArg::from_raw_parts(handle, LEN) };

    nested_lines::launch(
        &client,
        cube_count.clone(),
        CubeDim::new_1d(CUBE_DIM),
        buffer(input_handle.clone()),
        buffer(output_handle.clone()),
    );
    nested_full::launch(
        &client,
        cube_count,
        CubeDim::new_1d(CUBE_DIM),
        buffer(input_handle),
        buffer(output_handle.clone()),
    );

    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);
    println!("{:?}: output[3] = {}", client.name(), output[3]);
}
