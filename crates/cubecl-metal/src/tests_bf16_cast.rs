use cubecl_core::{self as cubecl, prelude::*};
use half::bf16;

type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn f32_to_bf16_cast(input: &[f32], output: &mut [bf16]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = bf16::cast_from(input[ABSOLUTE_POS]);
    }
}

#[test]
fn f32_to_bf16_cast_compiles_and_runs() {
    let client = R::client(&Default::default());
    let input = [1.0f32, -2.5, 3.25, 0.0];
    let len = input.len();

    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(len * core::mem::size_of::<bf16>());

    unsafe {
        f32_to_bf16_cast::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            BufferArg::from_raw_parts(input_handle, len),
            BufferArg::from_raw_parts(output_handle.clone(), len),
        );
    }

    let bytes = client.read_one_unchecked(output_handle);
    let actual = bf16::from_bytes(&bytes);

    for (actual, expected) in actual.iter().zip(input) {
        assert_eq!(*actual, bf16::from_f32(expected));
    }
}
