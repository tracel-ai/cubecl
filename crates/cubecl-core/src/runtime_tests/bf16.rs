//! The `bf16` conversions against the host type, bit for bit. A backend that carries `bf16` as a
//! 16-bit integer converts it in software, so the rounding and the special values are its own.

use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;
use half::bf16;
use std::println;

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::features::TypeUsage;

#[cube(launch_unchecked)]
fn kernel_decode<N: Size>(input: &[Vector<bf16, N>], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
fn kernel_encode<N: Size>(input: &[Vector<f32, N>], out: &mut [Vector<bf16, N>]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS]);
    }
}

fn supported(client: &Client) -> bool {
    let uses = bf16::supported_uses(client);
    uses.contains(TypeUsage::Conversion) && uses.contains(TypeUsage::Buffer)
}

/// Every `bf16` code widens to the `f32` the host type names. Widening is exact, so the
/// subnormals are kept too.
pub fn decode_exhaustive<R: Runtime>(client: Client, lanes: VectorSize) {
    if !supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let input = client.create_from_slice(u16::as_bytes(&codes));
    let out = client.empty(codes.len() * size_of::<f32>());
    let vectors = codes.len() / lanes;

    unsafe {
        kernel_decode::launch_unchecked(
            &client,
            CubeCount::Static(vectors.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            lanes,
            BufferArg::from_raw_parts(input, codes.len()),
            BufferArg::from_raw_parts(out.clone(), codes.len()),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);
    assert_eq!(
        actual.len(),
        codes.len(),
        "a failed launch reads back nothing"
    );
    for (code, actual) in codes.iter().zip(actual) {
        let expected = bf16::from_bits(*code).to_f32();
        // The host quiets a signalling NaN where a shift keeps it; only NaN-ness is portable.
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "bf16 {code:#06x} widens to a NaN, got {actual:e}"
            );
            continue;
        }
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "bf16 {code:#06x} widens to {expected:e}, got {actual:e}"
        );
    }
}

/// Narrowing rounds to nearest even, carries into the exponent and on to infinity, and keeps a
/// NaN a NaN however its payload sits.
pub fn encode<R: Runtime>(client: Client, lanes: VectorSize) {
    if !supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    let values: Vec<f32> = [
        0x3F80_8000, // 1 + half an ulp: a tie, stays on the even 1.0
        0x3F81_8000, // a tie above an odd code, rounds up
        0x3F80_8001, // just past a tie, rounds up
        0x3F80_7FFF, // just short of a tie, rounds down
        0x3FFF_C000, // a carry out of the mantissa into the exponent
        0x7F7F_FFFF, // f32::MAX rounds past the largest bf16, to infinity
        0x7F7F_7FFF, // the largest value that stays finite
        0x7F80_0000, // infinity
        0xFF80_0000, // negative infinity
        0x7FC0_0000, // the canonical NaN
        0x7F80_0001, // a NaN whose payload is only in the dropped bits
        0xFF80_8000, // a negative NaN rounding would carry into the sign
        0x0000_0000, // zero
        0x8000_0000, // negative zero keeps its sign
        0xC2F7_1234, // an ordinary negative value
        0x4049_0FDB, // pi
    ]
    .into_iter()
    .map(f32::from_bits)
    .collect();
    let input = client.create_from_slice(f32::as_bytes(&values));
    let out = client.empty(values.len() * size_of::<u16>());

    unsafe {
        kernel_encode::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d((values.len() / lanes) as u32),
            lanes,
            BufferArg::from_raw_parts(input, values.len()),
            BufferArg::from_raw_parts(out.clone(), values.len()),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = u16::from_bytes(&actual);
    assert_eq!(
        actual.len(),
        values.len(),
        "a failed launch reads back nothing"
    );
    for (value, actual) in values.iter().zip(actual) {
        let expected = bf16::from_f32(*value);
        let actual = bf16::from_bits(*actual);
        // Native converters pick their own NaN payload; only NaN-ness is portable.
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "{:#010x} narrows to a NaN, got {:#06x}",
                value.to_bits(),
                actual.to_bits()
            );
            continue;
        }
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{:#010x} narrows to {:#06x}",
            value.to_bits(),
            expected.to_bits()
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_bf16 {
    () => {
        mod bf16_conversion {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn decode_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                for lanes in [1, 4] {
                    cubecl_core::runtime_tests::bf16::decode_exhaustive::<TestRuntime>(
                        client.clone(),
                        lanes,
                    );
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn encode() {
                let client = TestRuntime::client(&Default::default());
                for lanes in [1, 4] {
                    cubecl_core::runtime_tests::bf16::encode::<TestRuntime>(client.clone(), lanes);
                }
            }
        }
    };
}
