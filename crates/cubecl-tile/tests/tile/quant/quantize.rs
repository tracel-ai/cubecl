use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use crate::tile::uncut;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl::{bytes::Bytes, prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_common::{e2m1, e4m3, ue8m0};
use cubecl_tile::{
    Axis, Launcher, Levels, Partitioning, Projection, Space, TileArg, TileArgLaunch, TileSpec,
    kind::Field,
    launch::Grid,
    layout::PhysicalAxisMap,
    space::{CubeOrder, PlaneOrder},
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

/// Each plane of the launch writing its box of the packed values from the float ones.
#[cube(launch)]
fn quantize_planes(input: &TileArg<'_>, packed: &TileArg<'_>, partitioning: Partitioning) {
    let input = input.tile(&partitioning);
    let packed = packed.tile(&partitioning);
    for plane in &partitioning {
        let mut packed = packed.at(&plane);
        packed.quantize_from(&input.at(&plane));
    }
}

/// How the float values are bound: as the words are, both block axes one dim of the row, or a
/// dim an axis.
#[derive(Clone, Copy)]
enum Source {
    AsTheWords,
    PerAxis,
}

/// Which workers copy: one unit over the whole space, or a plane a row, its units taking the
/// row's scales in turns, `planes` planes a cube, writing `words` words a load.
#[derive(Clone, Copy)]
enum Copy {
    OneUnit,
    PlaneARow { planes: usize, words: usize },
}

/// How the block scales are stored: plain `f32`s, or packed into words as `dtype`, as many to a
/// word as it holds.
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

/// `words` bound as the values they pack: a packed operand is declared in values.
fn in_values(words: &TensorHandle, rows: usize, cols: usize) -> TensorArg {
    let mut binding = words.clone().binding();
    binding.shape = vec![rows, cols].into();
    binding.strides = vec![cols, 1].into();
    binding.into_tensor_arg()
}

/// The low `bits` of a word.
fn mask(bits: usize) -> u32 {
    ((1u64 << bits) - 1) as u32
}

/// The scale a code of a packed word of scales stores.
fn host_scale_value(code: u32, dtype: ScaleDtype) -> f32 {
    match dtype {
        ScaleDtype::UE8M0 => ue8m0::from_bits(code as u8).to_f32(),
        ScaleDtype::UE4M3 => e4m3::from_bits(code as u8).to_f32(),
        ScaleDtype::F32 => f32::from_bits(code),
        ScaleDtype::F16 => half::f16::from_bits(code as u16).to_f32(),
        ScaleDtype::BF16 => half::bf16::from_bits(code as u16).to_f32(),
    }
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

/// `rows` rows of 160 blocks of `block` values each, quantized to `value` under `scales`, and
/// under a per-tensor `global` where there is one: the words and the scales the copy writes match
/// burn's calibration code for code. An all-zero block, whose scale is zero, is written as zeros.
fn check_quantize(value: QuantValue, block: usize, scales: Scales, global: Option<f32>) {
    check_quantize_as(
        value,
        block,
        scales,
        global,
        Source::AsTheWords,
        Copy::OneUnit,
    );
}

/// [`check_quantize`], the source bound as `source` says and copied as `copy` says.
fn check_quantize_as(
    value: QuantValue,
    block: usize,
    scales: Scales,
    global: Option<f32>,
    source: Source,
    copy: Copy,
) {
    let client = cubecl::test_device().client();
    let (rows, blocks) = (4, 160);
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
            let per_word = Field::of_scale(dtype).per_word();
            let t = TensorHandle::new_contiguous(
                vec![rows, blocks / per_word],
                client.create(Bytes::from_elems(vec![0u32; rows * blocks / per_word])),
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
    let words_a_load = match copy {
        Copy::OneUnit => 1,
        Copy::PlaneARow { words, .. } => words,
    };
    let values = TileArgLaunch::new(
        in_values(&words, rows, cols),
        u32::elem_type_native(),
        words_a_load,
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
    let input_arg = match source {
        Source::AsTheWords => TileArgLaunch::new(
            input_t.binding().into_tensor_arg(),
            f32::elem_type_native(),
            width,
            TileSpec::new(projection),
        ),
        Source::PerAxis => {
            let mut binding = input_t.binding();
            binding.shape = vec![rows, blocks, block].into();
            binding.strides = vec![cols, block, 1].into();
            TileArgLaunch::new(
                binding.into_tensor_arg(),
                f32::elem_type_native(),
                width,
                TileSpec::direct(&[M, KB, KI]),
            )
        }
    };
    match copy {
        Copy::OneUnit => quantize_copy::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            input_arg,
            values,
            uncut(&client, &space, &space).partitioning_arg(),
        ),
        Copy::PlaneARow { planes, .. } => {
            let order = PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(planes);
            let levels = Levels::leaf(&[(M, 1), (KB, blocks), (KI, block)])
                .launch_planes(&[M], order)
                .build();
            let launcher = Launcher::new(
                &client,
                Partitioning::new(space.clone(), levels),
                &space,
                Grid::FromLevels,
            )
            .unwrap();
            quantize_planes::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                input_arg,
                values,
                launcher.partitioning_arg(),
            )
        }
    }

    let got_scales: Vec<f32> = match scales {
        Scales::Plain => {
            let got = HostData::from_tensor_handle(&client, scales_t, HostDataType::F32);
            (0..rows * blocks)
                .map(|i| got.get_f32(&[i / blocks, i % blocks]))
                .collect()
        }
        Scales::Packed(dtype) => {
            let bits = Field::of_scale(dtype).size_bits();
            let got = client.read_one(scales_t.handle).unwrap().to_vec();
            let got: Vec<u32> = bytemuck::cast_slice(&got).to_vec();
            got.iter()
                .flat_map(|&word| (0..32 / bits).map(move |i| (word >> (i * bits)) & mask(bits)))
                .map(|code| host_scale_value(code, dtype))
                .collect()
        }
    };
    let mut want_codes = Vec::new();
    for (i, block_values) in input.chunks(block).enumerate() {
        let want = host_scale(block_values, value, dtype, global);
        let have = got_scales[i];
        // The device's division is the fast one: an `f32` scale, which nothing rounds after it,
        // may land an ulp from the host's.
        let ulps = (have.to_bits() as i64 - want.to_bits() as i64).abs();
        let allowed = match dtype {
            ScaleDtype::F32 => 1,
            _ => 0,
        };
        assert!(
            ulps <= allowed,
            "{value:?} scale {i}: {have} against {want}"
        );
        let effective = have * global.unwrap_or(1.0);
        want_codes.extend(block_values.iter().map(|&v| match effective {
            0.0 => 0,
            _ => host_code(v / effective, value),
        }));
    }
    let got_words = client.read_one(words.handle).unwrap().to_vec();
    let got_words: Vec<u32> = bytemuck::cast_slice(&got_words).to_vec();
    assert_eq!(got_words, host_words(&want_codes, bits), "{value:?} values");
}

/// `q8` under one `f32` scale a block of sixteen: the integer formats round to the nearest
/// whole number and clamp.
#[test]
fn q8_quantizes_as_burn_calibrates_it() {
    check_quantize(QuantValue::Q8S, 16, Scales::Plain, None);
}

/// `q8` under `f32` scales bound as words, as a packed tensor's scales are.
#[test]
fn q8_quantizes_under_scales_in_words() {
    check_quantize(QuantValue::Q8S, 16, Scales::Packed(ScaleDtype::F32), None);
}

/// `q4` under `f16` scales, two to a word: each rounded up onto the `f16` grid.
#[test]
fn q4_quantizes_under_half_scales() {
    check_quantize(QuantValue::Q4S, 32, Scales::Packed(ScaleDtype::F16), None);
}

/// `q8` under `bf16` scales, two to a word.
#[test]
fn q8_quantizes_under_bf16_scales() {
    check_quantize(QuantValue::Q8S, 32, Scales::Packed(ScaleDtype::BF16), None);
}

/// MXFP8 from a source bound a dim an axis, as a tensor restated over the block axes is.
#[test]
fn mxfp8_quantizes_from_a_source_per_axis() {
    check_quantize_as(
        QuantValue::E4M3,
        32,
        Scales::Packed(ScaleDtype::UE8M0),
        None,
        Source::PerAxis,
        Copy::OneUnit,
    );
}

/// NVFP4 copied a plane a row, its units taking the row's words of scales in turns, and the
/// planes of one cube each copying their own row, a word or a block's two words a load.
#[test]
fn nvfp4_quantizes_a_plane_a_row() {
    for (planes, words) in [(1, 1), (4, 1), (4, 2)] {
        check_quantize_as(
            QuantValue::E2M1,
            16,
            Scales::Packed(ScaleDtype::UE4M3),
            Some(0.01),
            Source::PerAxis,
            Copy::PlaneARow { planes, words },
        );
    }
}

/// MXFP8 copied a plane a row, four words a load.
#[test]
fn mxfp8_quantizes_four_words_a_load() {
    check_quantize_as(
        QuantValue::E4M3,
        32,
        Scales::Packed(ScaleDtype::UE8M0),
        None,
        Source::PerAxis,
        Copy::PlaneARow {
            planes: 2,
            words: 4,
        },
    );
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
