use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::quant::fp4::{
    e2m1_bits_to_float, e2m1_decode_host, e2m1_encode_host, e2m1_packed_bits_to_float,
    float_to_e2m1_bits,
};

#[cube(launch_unchecked)]
fn kernel_decode<N: Size>(codes: &[Vector<u32, N>], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < codes.len() {
        out[ABSOLUTE_POS] = e2m1_bits_to_float::<f32, N>(codes[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
fn kernel_encode<N: Size>(values: &[Vector<f32, N>], out: &mut [Vector<u32, N>]) {
    if ABSOLUTE_POS < values.len() {
        out[ABSOLUTE_POS] = float_to_e2m1_bits::<f32, N>(values[ABSOLUTE_POS]);
    }
}

/// The packed decoder [`crate::quant::dequantize`] actually calls: one word in, `N` codes out.
#[cube(launch_unchecked)]
fn kernel_decode_packed<N: Size>(words: &[u32], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < words.len() {
        out[ABSOLUTE_POS] = e2m1_packed_bits_to_float::<f32, N>(words[ABSOLUTE_POS]);
    }
}

/// The kernel codec has to agree with [`e2m1_decode_host`] and [`e2m1_encode_host`] exactly.
///
/// The module's own unit tests check the host pair against itself and against
/// `cubecl_common::e2m1`, which pins the specification but says nothing about the arithmetic that
/// runs on device. Only a
/// differential check does, and it is the whole point of the software path: it stands in for a
/// CUDA intrinsic, so it has to produce what that intrinsic's format produces.
///
/// No capability gate. The kernel names no 4-bit type — codes ride in `u32` and values in `f32` —
/// which is exactly the property that lets a backend with no `e2m1` decode one.
pub fn test_e2m1_codec_matches_host<R: Runtime>(client: ComputeClient<R>) {
    // Every code, then every code again under garbage upper bits: a caller may hand over an
    // unmasked field, and the decoder promises to ignore what is above the nibble.
    let codes: Vec<u32> = (0..16u32).chain((0..16u32).map(|c| c | 0xFFF0)).collect();
    let decoded = launch_decode::<R>(&client, &codes);

    let mut bad = vec![];
    for (i, &code) in codes.iter().enumerate() {
        let expected = e2m1_decode_host(code & 0xF);
        // Bit equality, not approximate: `-0.0` and `0.0` are the same number under `==`, and
        // telling them apart is the point of half these codes.
        if decoded[i].to_bits() != expected.to_bits() {
            bad.push(format!(
                "  decode {code:#x}: device {}, host {expected}",
                decoded[i]
            ));
        }
    }

    let values = encode_inputs();
    let encoded = launch_encode::<R>(&client, &values);
    for (i, &value) in values.iter().enumerate() {
        let expected = e2m1_encode_host(value);
        if encoded[i] != expected {
            bad.push(format!(
                "  encode {value:e}: device {:#x}, host {expected:#x}",
                encoded[i]
            ));
        }
    }

    // The packed decoder against the same reference, one nibble at a time. This is the entry
    // point `cast_masked_plain` reaches for, so it is the one a quantized read depends on.
    let words: Vec<u32> = (0..=0xFFu32).collect();
    let packed = launch_decode_packed::<R>(&client, &words);
    for (i, &word) in words.iter().enumerate() {
        for lane in 0..2 {
            let expected = e2m1_decode_host((word >> (4 * lane)) & 0xF);
            let actual = packed[2 * i + lane as usize];
            if actual.to_bits() != expected.to_bits() {
                bad.push(format!(
                    "  packed {word:#04x} lane {lane}: device {actual}, host {expected}"
                ));
            }
        }
    }

    if !bad.is_empty() {
        panic!(
            "{} disagree with the host codec\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}

/// The inputs the encoder is checked on: every value it can produce, every boundary it decides on,
/// and the ends it saturates.
fn encode_inputs() -> Vec<f32> {
    let mut values = vec![];
    // Every magnitude the format holds, both signs — these have to land back on their own code.
    for code in 0..16u32 {
        values.push(e2m1_decode_host(code));
    }
    // Every midpoint, and a step to either side of it. The midpoints are the ties, where the
    // strict/non-strict alternation is the only thing choosing a code.
    for midpoint in [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] {
        for value in [midpoint, midpoint * 0.999, midpoint * 1.001] {
            values.push(value);
            values.push(-value);
        }
    }
    // Past the top magnitude, where the count of cleared midpoints saturates rather than wrapping.
    for value in [6.1f32, 100.0, f32::MAX, f32::INFINITY] {
        values.push(value);
        values.push(-value);
    }
    // `f32::NAN` only, not its negation: a device is free to canonicalize a NaN's sign bit, and
    // this is a codec test rather than a test of what the hardware does to a payload.
    values.push(f32::NAN);
    values
}

fn launch_decode<R: Runtime>(client: &ComputeClient<R>, codes: &[u32]) -> Vec<f32> {
    let handle_in = client.create_from_slice(u32::as_bytes(codes));
    let handle_out = client.empty(size_of_val(codes));

    // WebGPU only guarantees 256 units per cube, so spread the values over cubes instead.
    let cube_dim = 256u32;

    unsafe {
        kernel_decode::launch_unchecked::<R>(
            client,
            CubeCount::Static((codes.len() as u32).div_ceil(cube_dim), 1, 1),
            CubeDim::new_1d(cube_dim),
            1,
            BufferArg::from_raw_parts(handle_in, codes.len()),
            BufferArg::from_raw_parts(handle_out.clone(), codes.len()),
        )
    };

    f32::from_bytes(&client.read_one_unchecked(handle_out)).to_vec()
}

fn launch_encode<R: Runtime>(client: &ComputeClient<R>, values: &[f32]) -> Vec<u32> {
    let handle_in = client.create_from_slice(f32::as_bytes(values));
    let handle_out = client.empty(size_of_val(values));

    let cube_dim = 256u32;

    unsafe {
        kernel_encode::launch_unchecked::<R>(
            client,
            CubeCount::Static((values.len() as u32).div_ceil(cube_dim), 1, 1),
            CubeDim::new_1d(cube_dim),
            1,
            BufferArg::from_raw_parts(handle_in, values.len()),
            BufferArg::from_raw_parts(handle_out.clone(), values.len()),
        )
    };

    u32::from_bytes(&client.read_one_unchecked(handle_out)).to_vec()
}

/// Returns two floats per word, the low nibble first.
fn launch_decode_packed<R: Runtime>(client: &ComputeClient<R>, words: &[u32]) -> Vec<f32> {
    let lanes = 2;
    let handle_in = client.create_from_slice(u32::as_bytes(words));
    let handle_out = client.empty(words.len() * lanes * size_of::<f32>());

    let cube_dim = 256u32;

    unsafe {
        kernel_decode_packed::launch_unchecked::<R>(
            client,
            CubeCount::Static((words.len() as u32).div_ceil(cube_dim), 1, 1),
            CubeDim::new_1d(cube_dim),
            lanes,
            BufferArg::from_raw_parts(handle_in, words.len()),
            BufferArg::from_raw_parts(handle_out.clone(), words.len() * lanes),
        )
    };

    f32::from_bytes(&client.read_one_unchecked(handle_out)).to_vec()
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_fp4 {
    () => {
        mod fp4 {
            use super::*;

            #[$crate::tests::test_log::test]
            fn e2m1_codec_matches_host() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::fp4::test_e2m1_codec_matches_host::<TestRuntime>(client);
            }
        }
    };
}
