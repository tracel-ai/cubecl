use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use crate::tile::uncut;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl::{bytes::Bytes, prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_common::{e2m1, e4m3, ue8m0};
use cubecl_tile::{
    Axis, Partitioning, Projection, Space, TileArg, TileArgLaunch, TileSpec, kind::Field,
    layout::PhysicalAxisMap,
};

const M: Axis = Axis(0);
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

/// The values packed from `input`, under the block scales they carry, which the copy computes.
#[cube(launch)]
fn quantize_copy(input: &TileArg<'_>, packed: &TileArg<'_>, space: Partitioning) {
    let input = input.tile(&space);
    let mut packed = packed.tile(&space);
    packed.quantize_from(&input);
}

/// How the block scales are stored: plain `f32`s, or packed four to a word as `dtype` codes.
#[derive(Clone, Copy)]
enum Scales {
    Plain,
    Packed(ScaleDtype),
}

/// A block's scale as burn calibrates it: the block's largest magnitude over the format's,
/// relative to `global` where there is one, rounded up to the scale's dtype.
fn host_scale(block: &[f32], value: QuantValue, dtype: ScaleDtype, global: Option<f32>) -> f32 {
    let (low, high) = value.range();
    let most = block.iter().fold(0.0f32, |most, v| most.max(v.abs()));
    let wanted = most / ((high - low) / 2.0);
    dtype.round_up(match global {
        Some(global) => wanted / global,
        None => wanted,
    })
}

/// `value` over its effective scale, as the code `format` stores it.
fn host_code(value: f32, format: QuantValue) -> u32 {
    let (low, high) = format.range();
    match format {
        QuantValue::E2M1 => e2m1::from_f32(value.clamp(low, high)).to_bits() as u32,
        QuantValue::E4M3 => e4m3::from_f32(value.clamp(low, high)).to_bits() as u32,
        _ => (value.round().clamp(low, high) as i32 as u32) & ((1 << format.size_bits()) - 1),
    }
}

/// A scale's code in a packed word of scales.
fn host_scale_code(scale: f32, dtype: ScaleDtype) -> u32 {
    match dtype {
        ScaleDtype::UE8M0 => ue8m0::from_f32(scale).to_bits() as u32,
        ScaleDtype::UE4M3 => e4m3::from_f32(scale).to_bits() as u32,
        other => panic!("a packed scale is an 8-bit code, not {other:?}"),
    }
}

/// `words` bound as the values they pack: a packed operand is declared in values.
fn in_values(words: &TensorHandle, rows: usize, cols: usize) -> TensorArg {
    let mut binding = words.clone().binding();
    binding.shape = vec![rows, cols].into();
    binding.strides = vec![cols, 1].into();
    binding.into_tensor_arg()
}

/// Codes `bits` wide packed into words, the first in each word's low bits.
fn host_words(codes: &[u32], bits: usize) -> Vec<u32> {
    codes
        .chunks(32 / bits)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0, |acc, (i, &code)| acc | (code << (i * bits)))
        })
        .collect()
}

/// `rows` rows of four blocks of `block` values each, quantized to `value` under `scales`, and
/// under a per-tensor `global` where there is one: the words and the scales the copy writes match
/// burn's calibration code for code. An all-zero block, whose scale is zero, is written as zeros.
fn check_quantize(value: QuantValue, block: usize, scales: Scales, global: Option<f32>) {
    let client = cubecl::test_device().client();
    let (rows, blocks) = (4, 4);
    let cols = block * blocks;
    let bits = value.size_bits();
    let per_word = 32 / bits;
    let width = per_word.min(client.properties().hardware.max_vector_size as usize);
    if width < 1 {
        ValidationResult::Skipped("no vectors".into()).enforce();
        return;
    }

    // Values over several magnitudes, signs and a zero block, deterministic.
    let mut seed = 0x2545f491u32;
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
            match i / block % 7 {
                3 => 0.0,
                band => unit * (1 << band) as f32,
            }
        })
        .collect();
    let (input_t, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(f32::elem_type_native())
        .custom(input.clone())
        .generate_with_f32_host_data();
    let words = TensorHandle::new_contiguous(
        vec![rows, cols / per_word],
        client.create(Bytes::from_elems(vec![0u32; rows * cols / per_word])),
        u32::elem_type_native(),
    );
    let space = Space::new(&[(M, rows), (KB, blocks), (KI, block)]);
    let projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    let (scales_t, scales_arg, dtype) = match scales {
        Scales::Plain => {
            let t = TestInput::builder(client.clone(), shape![rows, blocks])
                .dtype(f32::elem_type_native())
                .zeros()
                .generate_without_host_data();
            let arg = TileArgLaunch::new(
                t.clone().binding().into_tensor_arg(),
                f32::elem_type_native(),
                1,
                TileSpec::new(projection.scales_per(KB)),
            );
            (t, arg, ScaleDtype::F32)
        }
        Scales::Packed(dtype) => {
            let t = TensorHandle::new_contiguous(
                vec![rows, blocks / 4],
                client.create(Bytes::from_elems(vec![0u32; rows * blocks / 4])),
                u32::elem_type_native(),
            );
            let arg = TileArgLaunch::new(
                in_values(&t, rows, blocks),
                u32::elem_type_native(),
                1,
                TileSpec::new(projection.scales_per(KB)).packed(Field::of_scale(dtype)),
            );
            (t, arg, dtype)
        }
    };
    let values = TileArgLaunch::new(
        in_values(&words, rows, cols),
        u32::elem_type_native(),
        1,
        TileSpec::new(projection.clone()).packed(Field::Quant(value)),
    )
    .scales(scales_arg);
    let values = match global {
        None => values,
        Some(g) => {
            let t = TestInput::builder(client.clone(), shape![1])
                .dtype(f32::elem_type_native())
                .custom(vec![g])
                .generate_without_host_data();
            let per_tensor = Projection::new(&[M, KB, KI], &[PhysicalAxisMap::broadcast()]);
            values.scales(TileArgLaunch::new(
                t.binding().into_tensor_arg(),
                f32::elem_type_native(),
                1,
                TileSpec::new(per_tensor),
            ))
        }
    };
    quantize_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input_t.binding().into_tensor_arg(),
            f32::elem_type_native(),
            width,
            TileSpec::new(projection),
        ),
        values,
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let mut want_scales = Vec::new();
    let mut want_codes = Vec::new();
    for row in input.chunks(cols) {
        for block_values in row.chunks(block) {
            let scale = host_scale(block_values, value, dtype, global);
            let effective = scale * global.unwrap_or(1.0);
            want_scales.push(scale);
            want_codes.extend(block_values.iter().map(|&v| match effective {
                0.0 => 0,
                _ => host_code(v / effective, value),
            }));
        }
    }
    let got_words = client.read_one(words.handle).unwrap().to_vec();
    let got_words: Vec<u32> = bytemuck::cast_slice(&got_words).to_vec();
    assert_eq!(got_words, host_words(&want_codes, bits), "{value:?} values");
    match scales {
        Scales::Plain => {
            let got = HostData::from_tensor_handle(&client, scales_t, HostDataType::F32);
            for (i, &want) in want_scales.iter().enumerate() {
                let have = got.get_f32(&[i / blocks, i % blocks]);
                assert_eq!(have, want, "{value:?} scale {i}");
            }
        }
        Scales::Packed(dtype) => {
            let codes: Vec<u32> = want_scales
                .iter()
                .map(|&scale| host_scale_code(scale, dtype))
                .collect();
            let got = client.read_one(scales_t.handle).unwrap().to_vec();
            let got: Vec<u32> = bytemuck::cast_slice(&got).to_vec();
            assert_eq!(got, host_words(&codes, 8), "{value:?} scales");
        }
    }
}

/// `q8` under one `f32` scale a block of sixteen: the integer formats round to the nearest
/// whole number and clamp.
#[test]
fn q8_quantizes_as_burn_calibrates_it() {
    check_quantize(QuantValue::Q8S, 16, Scales::Plain, None);
}

/// MXFP8: `e4m3` values under `ue8m0` scales, four to a word, a block of thirty-two.
#[test]
fn mxfp8_quantizes_as_burn_calibrates_it() {
    check_quantize(
        QuantValue::E4M3,
        32,
        Scales::Packed(ScaleDtype::UE8M0),
        None,
    );
}

/// MXFP4: `e2m1` values, eight to a word, under `ue8m0` scales.
#[test]
fn mxfp4_quantizes_as_burn_calibrates_it() {
    check_quantize(
        QuantValue::E2M1,
        32,
        Scales::Packed(ScaleDtype::UE8M0),
        None,
    );
}

/// NVFP4: `e2m1` values under `ue4m3` scales of sixteen, relative to the per-tensor scale the
/// caller wrote.
#[test]
fn nvfp4_quantizes_under_its_per_tensor_scale() {
    check_quantize(
        QuantValue::E2M1,
        16,
        Scales::Packed(ScaleDtype::UE4M3),
        Some(0.01),
    );
}
