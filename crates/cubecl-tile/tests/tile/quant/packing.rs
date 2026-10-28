//! A field's codes in and out of words: every field reads back the values it packs.

use cubecl::e2m1x2;
use cubecl::ir::features::TypeUsage;
use cubecl::ir::{FloatKind, types::Fp8Format};
use cubecl::post_processing::fp4::E2M1_F16_LIFT;
use cubecl::post_processing::minifloat::fp8_f16_lift;
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantValue;
use cubecl_tile::kind::{Field, pack_line, unpack_line};

/// Two words of `field`'s codes packed from `values` and unpacked into `out`.
#[allow(clippy::needless_range_loop)]
#[cube(launch_unchecked)]
fn round_trip(values: &[f32], out: &mut [f32], #[comptime] field: Field) {
    let size!(NF) = comptime!(2 * field.per_word());
    let mut line = Vector::<f32, NF>::empty();
    #[unroll]
    for i in 0..NF::value() {
        line.insert(i, values[i]);
    }
    let words = pack_line::<Const<2>, NF>(line, field);
    let back = unpack_line::<f32, Const<2>, NF>(words, field);
    #[unroll]
    for i in 0..NF::value() {
        out[i] = back.extract(i);
    }
}

/// Values on `field`'s grid, both signs where it has them, its ends and a subnormal among them.
fn on_grid(field: Field) -> Vec<f32> {
    let e2m1 = vec![
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -6.0, -3.0,
    ];
    let e4m3 = vec![
        0.0,
        448.0,
        -448.0,
        0.015625,
        0.001953125,
        1.125,
        -3.5,
        240.0,
    ];
    match field {
        Field::Quant(QuantValue::Q8S) => vec![-128.0, -1.0, 0.0, 1.0, 37.0, 127.0, -77.0, 5.0],
        Field::Quant(QuantValue::Q4S) => (-8..8).map(|v| v as f32).collect(),
        Field::Quant(QuantValue::Q2S) => vec![-2.0, -1.0, 0.0, 1.0],
        Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1 => e2m1,
        Field::Placed {
            value: QuantValue::E2M1,
            ..
        } => e2m1,
        Field::Quant(QuantValue::E4M3) | Field::Fp8(Fp8Format::E4M3) => e4m3,
        Field::Placed {
            value: QuantValue::E4M3,
            ..
        } => e4m3,
        Field::Quant(QuantValue::E5M2) => {
            vec![
                0.0,
                57344.0,
                -2f32.powi(-16),
                1.5,
                -0.25,
                3.0,
                2f32.powi(-14),
                96.0,
            ]
        }
        Field::Fp8(Fp8Format::UE8M0) => {
            vec![
                1.0,
                2.0,
                0.5,
                2f32.powi(-100),
                2f32.powi(100),
                0.25,
                8.0,
                2f32.powi(-126),
            ]
        }
        Field::Float(FloatKind::F32) => vec![-3.25, 1.0e-30],
        // Normal, subnormal (2^-24, three of 2^-20's) and the largest finite.
        Field::Float(FloatKind::F16) => vec![
            271.0 / 4096.0,
            2f32.powi(-24),
            65504.0,
            -3.0 * 2f32.powi(-20),
        ],
        Field::Float(FloatKind::BF16) => vec![1.0, -2f32.powi(100), 0.0078125, 3.0],
        Field::Index { bits } => (0..1u32 << bits).map(|v| v as f32).collect(),
        other => panic!("no grid for {other:?}"),
    }
}

/// What a value of `field` reads back as: itself, or a lift short of it where it is read placed.
fn lift(field: Field) -> f32 {
    match field {
        Field::Placed {
            value: QuantValue::E2M1,
            lifted: false,
        } => E2M1_F16_LIFT as f32,
        Field::Placed {
            value: QuantValue::E4M3,
            lifted: false,
        } => fp8_f16_lift(Fp8Format::E4M3) as f32,
        _ => 1.0,
    }
}

/// `unpack_line(pack_line(x)) == x` for every field, two words of it.
#[test]
fn every_field_reads_back_what_it_packs() {
    let client = cubecl::test_device().client();
    let converts = e2m1x2::supported_uses(&client).contains(TypeUsage::Conversion);
    let mut fields = vec![
        Field::Quant(QuantValue::Q8S),
        Field::Quant(QuantValue::Q4S),
        Field::Quant(QuantValue::Q2S),
        Field::Quant(QuantValue::E2M1),
        Field::Quant(QuantValue::E4M3),
        Field::Quant(QuantValue::E5M2),
        Field::Fp8(Fp8Format::E4M3),
        Field::Fp8(Fp8Format::UE8M0),
        Field::Float(FloatKind::F32),
        Field::Float(FloatKind::F16),
        Field::Float(FloatKind::BF16),
        Field::Index { bits: 4 },
    ];
    for value in [QuantValue::E2M1, QuantValue::E4M3] {
        for lifted in [true, false] {
            fields.push(Field::Placed { value, lifted });
        }
    }
    if converts {
        fields.push(Field::ConvertedE2M1);
    }
    for field in fields {
        let count = 2 * field.per_word();
        let values: Vec<f32> = on_grid(field).into_iter().cycle().take(count).collect();
        let input = client.create_from_slice(f32::as_bytes(&values));
        let output = client.empty(count * size_of::<f32>());
        unsafe {
            round_trip::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_single(),
                BufferArg::from_raw_parts(input, count),
                BufferArg::from_raw_parts(output.clone(), count),
                field,
            )
        };
        let got = client.read_one_unchecked(output);
        let got = f32::from_bytes(&got);
        for (i, (&want, &have)) in values.iter().zip(got).enumerate() {
            assert_eq!(
                have * lift(field),
                want,
                "{field:?}: value {i} reads back {have}, packed from {want}"
            );
        }
    }
}
