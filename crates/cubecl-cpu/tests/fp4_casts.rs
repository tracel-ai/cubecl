//! `e2m1x2` casts as cubecl-llvm lowers them, onto the software codec (`LowerFp4CastPass`),
//! checked against the host type.
//!
//! The kernels cast directly rather than through the runtime tests, which skip a device that
//! reports no `e2m1x2` conversion: lowering a cast is the compiler's whatever the device reports,
//! and the CPU target is the one that runs it without a GPU.

use cubecl_common::{e2m1, e2m1x2};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

mod common;

#[cube(launch_unchecked)]
fn decode<N2: Size, N: Size>(input: &[Vector<e2m1x2, N2>], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
fn encode<N: Size, N2: Size>(input: &[Vector<f32, N>], out: &mut [Vector<e2m1x2, N2>]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS]);
    }
}

/// Every byte decodes to the two values the host type names, low nibble first, at every width:
/// under a word the codec decodes byte by byte, from a whole word four codes apart.
#[test]
fn every_byte_decodes_as_the_host_type() {
    let client = common::client_evaluating(None);
    let bytes: Vec<u8> = (0..=u8::MAX).collect();
    for bytes_per_vector in [1, 2, 4, 8] {
        let input = client.create_from_slice(&bytes);
        let out = client.empty(2 * bytes.len() * size_of::<f32>());
        let vectors = bytes.len() / bytes_per_vector;
        unsafe {
            decode::launch_unchecked(
                &client,
                CubeCount::Static(vectors.div_ceil(32) as u32, 1, 1),
                CubeDim::new_1d(32),
                bytes_per_vector,
                2 * bytes_per_vector,
                BufferArg::from_raw_parts(input, bytes.len()),
                BufferArg::from_raw_parts(out.clone(), 2 * bytes.len()),
            )
        };
        let actual = f32::from_bytes(&client.read_one_unchecked(out)).to_vec();
        assert_eq!(
            actual.len(),
            2 * bytes.len(),
            "a failed launch reads back nothing"
        );
        for (&byte, pair) in bytes.iter().zip(actual.chunks(2)) {
            let want = [byte & 0xF, byte >> 4].map(|code| e2m1::from_bits(code).to_f32());
            assert_eq!(
                pair.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                want.map(f32::to_bits),
                "byte {byte:#04x} at {bytes_per_vector} bytes a vector decoded to {pair:?}, \
                 want {want:?}"
            );
        }
    }
}

/// Values encode to the codes the host type rounds them to: every code point, every midpoint
/// between two (ties to even), past the top magnitude (saturating), and both zeros.
#[test]
fn values_encode_as_the_host_type() {
    let client = common::client_evaluating(None);
    let magnitudes = [
        0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
        100.0,
    ];
    let values: Vec<f32> = magnitudes
        .iter()
        .flat_map(|&magnitude| [magnitude, -magnitude])
        .collect();
    let bytes = values.len() / 2;
    for values_per_vector in [2, 8] {
        let input = client.create_from_slice(f32::as_bytes(&values));
        let out = client.empty(bytes);
        let vectors = values.len() / values_per_vector;
        unsafe {
            encode::launch_unchecked(
                &client,
                CubeCount::Static(vectors.div_ceil(32) as u32, 1, 1),
                CubeDim::new_1d(32),
                values_per_vector,
                values_per_vector / 2,
                BufferArg::from_raw_parts(input, values.len()),
                BufferArg::from_raw_parts(out.clone(), bytes),
            )
        };
        let actual = client.read_one_unchecked(out);
        for (pair, &byte) in values.chunks(2).zip(actual.iter()) {
            let want = e2m1::from_f32(pair[0]).to_bits() | (e2m1::from_f32(pair[1]).to_bits() << 4);
            assert_eq!(
                byte, want,
                "{pair:?} at {values_per_vector} values a vector encoded to {byte:#04x}, \
                 want {want:#04x}"
            );
        }
    }
    // The reference itself: the host packs pairs as the kernel does.
    let packed = e2m1x2::from_f32_slice(&values[..2]);
    assert_eq!(
        packed[0].to_bits(),
        e2m1::from_f32(values[0]).to_bits() | (e2m1::from_f32(values[1]).to_bits() << 4)
    );
}
