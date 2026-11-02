//! The packed-field decoders pinned to hand-computed values: low field first within a word,
//! word order in a line, sign extension, minifloat edge codes and table lookup.

use cubecl::prelude::*;
use cubecl_common::quant::scheme::{QuantMode, QuantScheme, QuantValue};
use cubecl_core::ir::types::Fp8Format;
use cubecl_core::post_processing::minifloat::{fp8_bits_to_f32, ue8m0_bits_to_f32};
use cubecl_core::{self as cubecl};
use cubecl_runtime::runtime::Runtime;

use crate::quant::{unpack_cast_u32, unpack_fields};

const NAN: f32 = f32::NAN;
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const TABLE: [f32; 16] = [
    -100.0, -10.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 8.0, 42.0,
];
/// Two words of codes and the eight values they hold, lowest byte of the first word first.
type Bytes8 = ([u32; 2], [f32; 8]);
const E4M3_CASES: [Bytes8; 2] = [
    // 0, -0, smallest subnormal 2^-9, smallest normal 2^-6, max 448, -max, NaN, -NaN.
    (
        [0x0801_8000, 0xFF7F_FE7E],
        [0.0, -0.0, 0.001953125, 0.015625, 448.0, -448.0, NAN, NAN],
    ),
    // 1, -1, 240, 2^-7, 256, 416, 0.5, -0.5.
    (
        [0x0477_B838, 0xB030_7D78],
        [1.0, -1.0, 240.0, 0.0078125, 256.0, 416.0, 0.5, -0.5],
    ),
];
const E5M2_CASES: [Bytes8; 2] = [
    // 0, -0, smallest subnormal 2^-16, smallest normal 2^-14, max 57344, -max, inf, -inf.
    (
        [0x0401_8000, 0xFC7C_FB7B],
        [
            0.0,
            -0.0,
            1.52587890625e-5,
            6.103515625e-5,
            57344.0,
            -57344.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
    ),
    // NaN, NaN, 1, -1, 2^-15, 3 * 2^-16, 2, 0.5.
    (
        [0xBC3C_7F7D, 0x3840_0302],
        [
            NAN,
            NAN,
            1.0,
            -1.0,
            3.0517578125e-5,
            4.57763671875e-5,
            2.0,
            0.5,
        ],
    ),
];
// 2^-125, 2^-126, 0.5, 1, 2, 2^127, NaN, 0.25.
const UE8M0_CASE: Bytes8 = (
    [0x7F7E_0102, 0x7DFF_FE80],
    [
        f32::from_bits(0x0100_0000),
        f32::from_bits(0x0080_0000),
        0.5,
        1.0,
        2.0,
        1.7014118e38,
        NAN,
        0.25,
    ],
);

/// Every NaN compares as one: its sign and payload are not the format's to promise.
fn bits(values: &[f32]) -> Vec<u32> {
    values
        .iter()
        .map(|v| {
            if v.is_nan() {
                NAN.to_bits()
            } else {
                v.to_bits()
            }
        })
        .collect()
}

#[cube(launch_unchecked)]
fn kernel_unpack(
    words: &[u32],
    table: ComptimeOption<Box<[f32]>>,
    out: &mut [f32],
    #[comptime] scheme: QuantScheme,
) {
    let size!(NF) = comptime!(2 * scheme.num_quants());
    let mut line = Vector::<u32, Const<2>>::empty();
    line.insert(0usize, words[0]);
    line.insert(1usize, words[1]);
    let values = unpack_cast_u32::<f32, Const<2>, NF>(line, table, scheme);
    #[unroll]
    for i in 0..NF::value() {
        out[i] = values.extract(i);
    }
}

/// `args` is the word, then the first field.
#[cube(launch_unchecked)]
fn kernel_fields(
    args: &[u32],
    table: ComptimeOption<Box<[f32]>>,
    out: &mut [f32],
    #[comptime] width: usize,
    #[comptime] scheme: QuantScheme,
) {
    let size!(NF) = comptime!(width);
    let values = unpack_fields::<f32, NF>(args[0], args[1], table, scheme);
    #[unroll]
    for i in 0..NF::value() {
        out[i] = values.extract(i);
    }
}

#[cube(launch_unchecked)]
fn kernel_bytes(codes: &[u32], out: &mut [f32], #[comptime] format: Fp8Format) {
    if ABSOLUTE_POS < codes.len() {
        let code = Vector::<u32, Const<1>>::new(codes[ABSOLUTE_POS]);
        let value = match comptime!(format) {
            Fp8Format::UE8M0 => ue8m0_bits_to_f32::<Const<1>>(code),
            _ => fp8_bits_to_f32::<Const<1>>(code, format),
        };
        out[ABSOLUTE_POS] = value.extract(0usize);
    }
}

fn table_arg(client: &Client, table: Option<&[f32]>) -> Option<BufferArg> {
    table.map(|t| unsafe {
        BufferArg::from_raw_parts(client.create_from_slice(f32::as_bytes(t)), t.len())
    })
}

fn unpack(
    client: &Client,
    words: [u32; 2],
    table: Option<&[f32]>,
    scheme: QuantScheme,
) -> Vec<f32> {
    let count = 2 * scheme.num_quants();
    let input = client.create_from_slice(u32::as_bytes(&words));
    let output = client.empty(count * size_of::<f32>());
    unsafe {
        kernel_unpack::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            BufferArg::from_raw_parts(input, 2),
            table_arg(client, table).into(),
            BufferArg::from_raw_parts(output.clone(), count),
            scheme,
        )
    };
    f32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

fn fields(
    client: &Client,
    word: u32,
    first: u32,
    width: usize,
    table: Option<&[f32]>,
    scheme: QuantScheme,
) -> Vec<f32> {
    let input = client.create_from_slice(u32::as_bytes(&[word, first]));
    let output = client.empty(width * size_of::<f32>());
    unsafe {
        kernel_fields::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            BufferArg::from_raw_parts(input, 2),
            table_arg(client, table).into(),
            BufferArg::from_raw_parts(output.clone(), width),
            width,
            scheme,
        )
    };
    f32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

fn decode_bytes(client: &Client, codes: &[u32], format: Fp8Format) -> Vec<f32> {
    let input = client.create_from_slice(u32::as_bytes(codes));
    let output = client.empty(codes.len() * size_of::<f32>());
    unsafe {
        kernel_bytes::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(codes.len() as u32),
            BufferArg::from_raw_parts(input, codes.len()),
            BufferArg::from_raw_parts(output.clone(), codes.len()),
            format,
        )
    };
    f32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

fn scheme(value: QuantValue) -> QuantScheme {
    QuantScheme::default().with_value(value)
}

/// Q8 and Q4 and Q2, the signed and the full-range variants alike: the top bit of a slot is its
/// sign, the lowest field of a word comes first and the first word before the second.
pub fn test_integer_fields_sign_extend<R: Runtime>(client: Client) {
    let q2_a = [-1.0, -2.0, 1.0, 0.0].repeat(4);
    let q2_b = [0.0, 1.0, -2.0, -1.0].repeat(4);
    let cases: [(&[QuantValue], [u32; 2], Vec<f32>); 3] = [
        (
            &[QuantValue::Q8S, QuantValue::Q8F],
            [0x80FF_017F, 0xC03F_FE00],
            vec![127.0, 1.0, -1.0, -128.0, 0.0, -2.0, 63.0, -64.0],
        ),
        (
            &[QuantValue::Q4S, QuantValue::Q4F],
            [0x7654_3210, 0xFEDC_BA98],
            (0..16)
                .map(|v: i32| (if v < 8 { v } else { v - 16 }) as f32)
                .collect(),
        ),
        (
            &[QuantValue::Q2S, QuantValue::Q2F],
            [0x1B1B_1B1B, 0xE4E4_E4E4],
            [q2_a, q2_b].concat(),
        ),
    ];
    for (values, words, expected) in cases {
        for &value in values {
            let actual = unpack(&client, words, None, scheme(value));
            assert_eq!(actual, expected, "{value:?}");
        }
    }
}

/// All sixteen codes, then again with the two nibbles of every byte swapped.
pub fn test_e2m1_both_nibble_orders<R: Runtime>(client: Client) {
    let swapped: Vec<f32> = E2M1.chunks(2).flat_map(|p| [p[1], p[0]]).collect();
    for (words, expected) in [
        ([0x7654_3210, 0xFEDC_BA98], E2M1.to_vec()),
        ([0x6745_2301, 0xEFCD_AB89], swapped),
    ] {
        let actual = unpack(&client, words, None, scheme(QuantValue::E2M1));
        assert_eq!(bits(&actual), bits(&expected));
    }
}

/// The software decoders on their edge codes, the format's own bits to `f32`.
pub fn test_fp8_and_ue8m0_edge_codes<R: Runtime>(client: Client) {
    let cases = [
        (Fp8Format::E4M3, E4M3_CASES.to_vec()),
        (Fp8Format::E5M2, E5M2_CASES.to_vec()),
        (Fp8Format::UE8M0, vec![UE8M0_CASE]),
    ];
    for (format, cases) in cases {
        for (words, expected) in cases {
            let codes: Vec<u32> = words
                .iter()
                .flat_map(|w| (0..4).map(move |j| (w >> (8 * j)) & 0xFF))
                .collect();
            let actual = decode_bytes(&client, &codes, format);
            assert_eq!(bits(&actual), bits(&expected), "{format:?} {words:x?}");
        }
    }
}

/// A lookup field is an index: the entry, not the index, and in field order.
pub fn test_table_lookup<R: Runtime>(client: Client) {
    let scheme = scheme(QuantValue::Q4F).with_mode(QuantMode::Lookup);
    let words = [0xFEDC_BA98, 0x7654_3210];
    let actual = unpack(&client, words, Some(&TABLE), scheme);
    assert_eq!(actual, [&TABLE[8..], &TABLE[..8]].concat());
}

/// A slice of a word starting at a runtime field, at either end of the word.
pub fn test_unpack_fields_slices_a_word<R: Runtime>(client: Client) {
    let q4 = scheme(QuantValue::Q4S);
    let q8 = scheme(QuantValue::Q8S);
    let lookup = scheme(QuantValue::Q4F).with_mode(QuantMode::Lookup);
    let read =
        |word, first, width, table, scheme| fields(&client, word, first, width, table, scheme);
    assert_eq!(read(0xFEDC_BA98, 2, 4, None, q4), [-6.0, -5.0, -4.0, -3.0]);
    assert_eq!(read(0xFEDC_BA98, 6, 2, None, q4), [-2.0, -1.0]);
    assert_eq!(read(0x80FF_017F, 1, 2, None, q8), [1.0, -1.0]);
    assert_eq!(read(0x80FF_017F, 2, 2, None, q8), [-1.0, -128.0]);
    assert_eq!(
        read(0xFEDC_BA98, 6, 2, Some(&TABLE), lookup),
        [TABLE[14], TABLE[15]]
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_packed_decode {
    () => {
        mod packed_decode {
            use super::*;

            #[$crate::tests::test_log::test]
            fn integer_fields_sign_extend() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::packed_decode::test_integer_fields_sign_extend::<TestRuntime>(
                    client,
                );
            }

            #[$crate::tests::test_log::test]
            fn e2m1_both_nibble_orders() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::packed_decode::test_e2m1_both_nibble_orders::<TestRuntime>(
                    client,
                );
            }

            #[$crate::tests::test_log::test]
            fn fp8_and_ue8m0_edge_codes() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::packed_decode::test_fp8_and_ue8m0_edge_codes::<TestRuntime>(
                    client,
                );
            }

            #[$crate::tests::test_log::test]
            fn table_lookup() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::packed_decode::test_table_lookup::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn unpack_fields_slices_a_word() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::packed_decode::test_unpack_fields_slices_a_word::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
