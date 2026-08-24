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
    // Cover exact values, rounding, signed zero, range extremes, subnormals, infinities and NaN.
    // NaN payloads are allowed to be canonicalized by the GPU, so they are compared by class
    // below; every non-NaN value is compared bit-for-bit with `half`'s reference conversion.
    let input = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        3.25,
        1.003_906_2, // Halfway between adjacent bf16 values around 1.0 (ties-to-even).
        1.003_906_4, // Immediately above that midpoint.
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
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

    let bytes = client
        .read_one(output_handle)
        .expect("the f32-to-bf16 Metal kernel should compile and execute");
    let actual = bf16::from_bytes(&bytes);

    for (actual, input) in actual.iter().zip(input) {
        let expected = bf16::from_f32(input);
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual:?}");
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "unexpected conversion for input {input:?}"
            );
        }
    }
}
