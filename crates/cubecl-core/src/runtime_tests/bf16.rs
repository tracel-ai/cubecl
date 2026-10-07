//! The `bf16` conversions against the host type, bit for bit. A backend that carries `bf16` as a
//! 16-bit integer converts it in software, so the rounding and the special values are its own.

use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;
use half::{bf16, f16};
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
            "{:#010x} narrows to {:#06x}, got {:#06x}",
            value.to_bits(),
            expected.to_bits(),
            actual.to_bits()
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_encode_from<S: Scalar>(input: &[S], out: &mut [bf16]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = bf16::cast_from(input[ABSOLUTE_POS]);
    }
}

/// Narrowing a source with more significant bits than `f32` rounds once. Each first value sits
/// just past a `bf16` tie that rounding to `f32` first would land on exactly, then round to even
/// the wrong way.
///
/// The expected codes are spelled out: `half::bf16::from_f64` goes through `f32` and rounds
/// twice itself.
pub fn encode_wide_sources<R: Runtime>(client: Client) {
    if !supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    // 2^31 + 2^23 + 2^7: half a `bf16` ulp past 2^31, plus a bit `f32` cannot hold.
    encode_from::<u32>(
        &client,
        &[
            (0x8080_0080, 0x4F01),
            (0x7FFF_FFFF, 0x4F00),
            (1, 0x3F80),
            (0, 0),
        ],
    );
    // -(2^30 + 2^22 + 2^6).
    encode_from::<i32>(
        &client,
        &[
            (-0x4040_0040, 0xCE81),
            (i32::MIN, 0xCF00),
            (-3, 0xC040),
            (0, 0),
        ],
    );
    // 1 + 2^-8 + 2^-52.
    let past_tie = f64::from_bits(0x3FF0_1000_0000_0001);
    encode_from::<f64>(
        &client,
        &[
            (past_tie, 0x3F81),
            (-past_tie, 0xBF81),
            (1e300, 0x7F80),
            (f64::MIN_POSITIVE, 0),
            (f64::INFINITY, 0x7F80),
            (f64::NEG_INFINITY, 0xFF80),
            (-0.0, 0x8000),
        ],
    );
    // 2^63 + 2^55 + 1.
    encode_from::<u64>(
        &client,
        &[
            ((1 << 63) + (1 << 55) + 1, 0x5F01),
            (u64::MAX, 0x5F80),
            (1 << 40, 0x5380),
            (0, 0),
        ],
    );
    // 2^62 + 2^54 + 1.
    let past_tie = (1i64 << 62) + (1 << 54) + 1;
    encode_from::<i64>(
        &client,
        &[
            (past_tie, 0x5E81),
            (-past_tie, 0xDE81),
            (i64::MIN, 0xDF00),
            (-1, 0xBF80),
        ],
    );
}

/// Narrows each value on the device and checks it against its expected `bf16` code.
fn encode_from<S: Scalar + CubeElement>(client: &Client, cases: &[(S, u16)]) {
    if !S::supported_uses(client).contains(TypeUsage::Conversion) {
        println!("Unsupported, skipping");
        return;
    }
    let values: Vec<S> = cases.iter().map(|(value, _)| *value).collect();
    let input = client.create_from_slice(S::as_bytes(&values));
    let out = client.empty(values.len() * size_of::<u16>());

    unsafe {
        kernel_encode_from::launch_unchecked::<S>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(values.len() as u32),
            BufferArg::from_raw_parts(input, values.len()),
            BufferArg::from_raw_parts(out.clone(), values.len()),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = u16::from_bytes(&actual);
    assert_eq!(
        actual.len(),
        cases.len(),
        "a failed launch reads back nothing"
    );
    for ((value, expected), actual) in cases.iter().zip(actual) {
        assert_eq!(
            actual, expected,
            "{value:?} narrows to {expected:#06x}, got {actual:#06x}"
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_convert<A: Scalar, B: Scalar>(input: &[A], out: &mut [B]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = B::cast_from(input[ABSOLUTE_POS]);
    }
}

/// Every `f16` code converts to the `bf16` nearest it and back, as through `f32`. The two share a
/// width but no conversion instruction, so a backend that only relabels the bits is caught here.
pub fn half_precision_exhaustive<R: Runtime>(client: Client) {
    let halves = f16::supported_uses(&client);
    if !supported(&client) || !halves.is_superset(TypeUsage::Conversion | TypeUsage::Buffer) {
        println!("Unsupported, skipping");
        return;
    }

    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let to_bf16 = convert::<f16, bf16>(&client, &codes);
    let to_f16 = convert::<bf16, f16>(&client, &codes);
    for (code, (to_bf16, to_f16)) in codes.iter().zip(to_bf16.into_iter().zip(to_f16)) {
        let expected = bf16::from_f32(f16::from_bits(*code).to_f32());
        let actual = bf16::from_bits(to_bf16);
        let case = Case::new("f16", *code);
        case.assert(
            actual.to_bits(),
            actual.is_nan(),
            expected.to_bits(),
            expected.is_nan(),
        );
        let expected = f16::from_f32(bf16::from_bits(*code).to_f32());
        let actual = f16::from_bits(to_f16);
        let case = Case::new("bf16", *code);
        case.assert(
            actual.to_bits(),
            actual.is_nan(),
            expected.to_bits(),
            expected.is_nan(),
        );
    }
}

/// `codes` read as `A` on the device and cast to `B`, as `B`'s bits.
fn convert<A: Scalar + CubeElement, B: Scalar + CubeElement>(
    client: &Client,
    codes: &[u16],
) -> Vec<u16> {
    let input = client.create_from_slice(u16::as_bytes(codes));
    let out = client.empty(core::mem::size_of_val(codes));
    unsafe {
        kernel_convert::launch_unchecked::<A, B>(
            client,
            CubeCount::Static(codes.len().div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(input, codes.len()),
            BufferArg::from_raw_parts(out.clone(), codes.len()),
        )
    };
    let actual = client.read_one_unchecked(out);
    let actual = u16::from_bytes(&actual).to_vec();
    assert_eq!(
        actual.len(),
        codes.len(),
        "a failed launch reads back nothing"
    );
    actual
}

/// One half-precision code converted to the other format.
#[derive(new)]
struct Case {
    from: &'static str,
    code: u16,
}

impl Case {
    /// The converted bits match, or are any NaN where a NaN is expected: the backends pick
    /// their own NaN payload.
    fn assert(&self, actual: u16, actual_is_nan: bool, expected: u16, expected_is_nan: bool) {
        let Self { from, code } = self;
        if expected_is_nan {
            assert!(
                actual_is_nan,
                "{from} {code:#06x} converts to a NaN, got {actual:#06x}"
            );
        } else {
            assert_eq!(
                actual, expected,
                "{from} {code:#06x} converts to {expected:#06x}, got {actual:#06x}"
            );
        }
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
            fn encode_wide_sources() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::bf16::encode_wide_sources::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn half_precision_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::bf16::half_precision_exhaustive::<TestRuntime>(client);
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
