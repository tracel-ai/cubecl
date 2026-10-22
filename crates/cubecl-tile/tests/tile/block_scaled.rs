//! `acc.mma(&a.mul(&sa), &b.mul(&sb))` over `e2m1` values under `e4m3` scales, one every sixteen
//! values of `k`: NVFP4, contracted through the device's block-scaled instruction, the values and
//! scales handed to it as they lie rather than decoded first.
//!
//! The kernel is the plain staged walk a float product takes, its instruction stating its operands
//! block-scaled (`MmaOperands::BlockScaled`). A per-tensor factor above the block scales rides an
//! operand whose block scales are words of four `e4m3`, which a stage keeps beside the values, so
//! the factor leaves for the sum. It runs only where the device offers that instruction, and is
//! held to the reference on codes and scales that are exact in `f32`.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, ir::FloatKind, prelude::*, quant::scheme::QuantValue, zspace::Shape};
use cubecl_common::{e2m1, e4m3};
use cubecl_tile::{
    Axis, Instruction, Level, Levels, MmaOperands, Partitioning, Projection, Semiring, Space,
    StageStorage, Stages, TileArg, TileArgLaunch, TileSpec, kind::Field, layout::PhysicalAxisMap,
    ops::matmul::MmaIo,
};

use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// `k` as the two axes a scale block makes of it: the block, and the value inside it.
const KB: Axis = Axis(3);
const KI: Axis = Axis(4);

/// Values one scale covers along `k`: NVFP4's block.
const BLOCK: usize = 16;

/// Values of `k` one instruction step covers, and so one word of four block scales.
const STEP: usize = 64;

/// `c = (a ⊗ sa) · (b ⊗ sb)`, a stage of `K` at a time, contracted through the manual-mma
/// instruction, the block scales applied to each stage as numbers.
#[cube(launch)]
fn block_scaled_matmul<E: Numeric>(
    a: &TileArg<'_>,
    a_scales: &TileArg<'_>,
    b: &TileArg<'_>,
    b_scales: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    let a_scales = a_scales.tile(&space);
    let b_scales = b_scales.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Mma {
            io: MmaIo::manual(),
            operands: MmaOperands::BlockScaled,
        }),
        Semiring::SUM_PROD,
    );
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        let a_scales_r = a_scales.at(region);
        let b_scales_r = b_scales.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(&a_s.mul(&a_scales_r), &b_s.mul(&b_scales_r));
        });
    });
    c.dequantize_from(&acc);
}

/// `c = (a ⊗ sa ⊗ ga) · (b ⊗ sb ⊗ gb)`: each operand carries its block scales as words of `e4m3`,
/// which a stage keeps beside the values, and a per-tensor factor above them, which the
/// instruction holds no level for, so it leaves the operand and multiplies the sum once, as it
/// drains.
#[cube(launch)]
fn block_scaled_matmul_under_words<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let instruction = comptime!(Instruction::Mma {
        io: MmaIo::manual(),
        operands: MmaOperands::BlockScaled,
    });
    let mut c = c.tile(&space);
    let (a, a_factor) = a.tile(&space).factor(c.space(), instruction);
    let (b, b_factor) = b.tile(&space).factor(c.space(), instruction);
    let acc = c.accumulator::<E>(&a, &b, instruction, Semiring::SUM_PROD);
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s);
        });
    });
    c.dequantize_from(&acc.scaled(&a_factor).scaled(&b_factor));
}

/// A `16 × 8` product of `k` deep, `a` row-major and `b` held `[n, k]`, against the host sum of
/// the decoded values under their scales, and under each operand's per-tensor factor where
/// `globals` states them. Skips where the device offers no block-scaled `e2m1` instruction, whose
/// path is the subject here.
fn check_block_scaled(k: usize, globals: Option<(f32, f32)>) {
    let client = cubecl::test_device().client();
    if !offers_nvfp4(&client) {
        ValidationResult::Skipped(
            "the device offers no block-scaled e2m1 instruction under e4m3 scales".to_string(),
        )
        .enforce();
        return;
    }
    let (m, n) = (16, 8);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, 64)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    // Every code of `e2m1`, varied by position so no row or column repeats another.
    let a_codes: Vec<u8> = (0..m * k).map(|i| ((i * 7 + i / k) % 16) as u8).collect();
    let b_codes: Vec<u8> = (0..n * k)
        .map(|i| ((i * 5 + 3 * (i / k)) % 16) as u8)
        .collect();
    // Scales exact in `e4m3`: a power of two times one of its mantissas.
    let scale = |i: usize| (1 + i % 8) as f32 * 2f32.powi((i % 5) as i32 - 3);
    let a_scale_values: Vec<f32> = (0..m * k / BLOCK).map(scale).collect();
    let b_scale_values: Vec<f32> = (0..n * k / BLOCK).map(|i| scale(i + 3)).collect();

    let words = |codes: &[u8]| -> Vec<u32> {
        codes
            .chunks(8)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &code)| acc | (code as u32) << (4 * j))
            })
            .collect()
    };
    let a_words = client.create_from_slice(u32::as_bytes(&words(&a_codes)));
    let b_words = client.create_from_slice(u32::as_bytes(&words(&b_codes)));
    let c = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();

    let field = Field::Quant(QuantValue::E2M1);
    // Values counted in values; their buffers hold words of eight along `k`.
    let values = |rows: Axis| TileSpec::new(Projection::direct(&[rows, K])).packed(field);
    let scales = |rows: Axis| {
        TileSpec::new(Projection::new(
            &[rows, K],
            &[
                PhysicalAxisMap::of(rows),
                PhysicalAxisMap::of(K).over(BLOCK),
            ],
        ))
    };
    let tensor = |handle, rows: usize, cols: usize| unsafe {
        TensorArg::from_raw_parts(handle, [cols, 1].into(), [rows, cols].into())
    };
    // The values are bound at the words their buffers hold, which the block-scaled instruction
    // reads as they lie.
    let stored = u32::elem_type_native();
    let element = f32::elem_type_native();
    // One `f32` over the whole operand, read as the word it is stored in.
    let global = |value: f32| {
        let word = client.create_from_slice(f32::as_bytes(&[value]));
        let spec = TileSpec::new(Projection::new(&[M, N, K], &[PhysicalAxisMap::broadcast()]))
            .packed(Field::Float(FloatKind::F32));
        TileArgLaunch::new(tensor(word, 1, 1), u32::elem_type_native(), 1, spec)
    };
    let out = TileArgLaunch::new(
        c.clone().binding().into_tensor_arg(),
        element,
        1,
        TileSpec::direct(&[M, N]),
    );
    match globals {
        // Block scales the values carry beside them only as words of four `e4m3`, as an NVFP4
        // checkpoint packs them: a factor above scales read as numbers is decoded into the values
        // as they are staged, which the block-scaled instruction cannot read.
        Some((a_global, b_global)) => {
            let scale_words = |values: &[f32]| -> Vec<u32> {
                values
                    .chunks(STEP / BLOCK)
                    .map(|word| {
                        word.iter().enumerate().fold(0u32, |acc, (j, &scale)| {
                            acc | (e4m3::from_f32(scale).to_bits() as u32) << (8 * j)
                        })
                    })
                    .collect()
            };
            let word_scales = |values: &[f32], rows: usize, axis: Axis| {
                let handle = client.create_from_slice(u32::as_bytes(&scale_words(values)));
                let spec = TileSpec::new(Projection::new(
                    &[axis, K],
                    &[PhysicalAxisMap::of(axis), PhysicalAxisMap::of(K).over(STEP)],
                ));
                TileArgLaunch::new(
                    tensor(handle, rows, k / STEP),
                    u32::elem_type_native(),
                    1,
                    spec,
                )
            };
            let operand = |words, rows: usize, axis: Axis, scales: &[f32], global_value: f32| {
                TileArgLaunch::new(tensor(words, rows, k), stored, 1, values(axis))
                    .scales(word_scales(scales, rows, axis))
                    .scales(global(global_value))
            };
            block_scaled_matmul_under_words::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                operand(a_words, m, M, &a_scale_values, a_global),
                operand(b_words, n, N, &b_scale_values, b_global),
                out,
                launcher.partitioning_arg(),
                launcher.partitioning().level(0),
                element,
            );
        }
        None => {
            let a_scales = client.create_from_slice(f32::as_bytes(&a_scale_values));
            let b_scales = client.create_from_slice(f32::as_bytes(&b_scale_values));
            block_scaled_matmul::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                TileArgLaunch::new(tensor(a_words, m, k), stored, 1, values(M)),
                TileArgLaunch::new(tensor(a_scales, m, k / BLOCK), element, 1, scales(M)),
                TileArgLaunch::new(tensor(b_words, n, k), stored, 1, values(N)),
                TileArgLaunch::new(tensor(b_scales, n, k / BLOCK), element, 1, scales(N)),
                out,
                launcher.partitioning_arg(),
                launcher.partitioning().level(0),
                element,
            );
        }
    }

    let decoded = |code: u8| e2m1::from_bits(code).to_f32();
    let (a_global, b_global) = globals.unwrap_or((1.0, 1.0));
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k)
                .map(|l| {
                    decoded(a_codes[i * k + l])
                        * a_scale_values[i * (k / BLOCK) + l / BLOCK]
                        * decoded(b_codes[j * k + l])
                        * b_scale_values[j * (k / BLOCK) + l / BLOCK]
                })
                .sum::<f32>()
                * a_global
                * b_global;
            let have = got.get_f32(&[i, j]);
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "k = {k}, element ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// [`check_block_scaled`] under per-tensor factors, `k` split into its blocks and the value inside
/// one (`KB·KI`), as a packed product states it: the values over both axes, their block scales
/// words of four `e4m3` scales, one a word of `KB` (`KB` divided by four), as the block-scaled
/// instruction reads them, and each step of the instruction four blocks deep, one word.
fn check_block_scaled_in_blocks(k: usize, (a_global, b_global): (f32, f32)) {
    let client = cubecl::test_device().client();
    if !offers_nvfp4(&client) {
        ValidationResult::Skipped(
            "the device offers no block-scaled e2m1 instruction under e4m3 scales".to_string(),
        )
        .enforce();
        return;
    }
    let (m, n, blocks) = (16, 8, k / BLOCK);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (KB, blocks), (KI, BLOCK)]),
            Levels::leaf(&[(M, m), (N, n), (KB, STEP / BLOCK), (KI, BLOCK)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    let a_codes: Vec<u8> = (0..m * k).map(|i| ((i * 7 + i / k) % 16) as u8).collect();
    let b_codes: Vec<u8> = (0..n * k)
        .map(|i| ((i * 5 + 3 * (i / k)) % 16) as u8)
        .collect();
    let scale = |i: usize| (1 + i % 8) as f32 * 2f32.powi((i % 5) as i32 - 3);
    let a_scale_values: Vec<f32> = (0..m * blocks).map(scale).collect();
    let b_scale_values: Vec<f32> = (0..n * blocks).map(|i| scale(i + 3)).collect();
    let pack = |fields: &[u32], bits: usize| -> Vec<u32> {
        fields
            .chunks(32 / bits)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &field)| acc | field << (bits * j))
            })
            .collect()
    };
    let codes = |codes: &[u8]| pack(&codes.iter().map(|&c| c as u32).collect::<Vec<_>>(), 4);
    let scale_words = |values: &[f32]| {
        let fields: Vec<u32> = values
            .iter()
            .map(|&v| e4m3::from_f32(v).to_bits() as u32)
            .collect();
        pack(&fields, 8)
    };
    let c = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();

    let tensor = |handle, rows: usize, cols: usize| unsafe {
        TensorArg::from_raw_parts(handle, [cols, 1].into(), [rows, cols].into())
    };
    let along_k = |rows: Axis| {
        Projection::new(
            &[rows, KB, KI],
            &[
                PhysicalAxisMap::of(rows),
                PhysicalAxisMap::disjoint(&[(KB, BLOCK), (KI, 1)]),
            ],
        )
    };
    let words = u32::elem_type_native();
    let element = f32::elem_type_native();
    let operand = |codes: Vec<u32>, scales: Vec<u32>, rows: usize, axis: Axis, global: f32| {
        let values = TileArgLaunch::new(
            tensor(client.create_from_slice(u32::as_bytes(&codes)), rows, k),
            words,
            1,
            TileSpec::new(along_k(axis)).packed(Field::Quant(QuantValue::E2M1)),
        );
        let block_scales = TileArgLaunch::new(
            tensor(
                client.create_from_slice(u32::as_bytes(&scales)),
                rows,
                k / STEP,
            ),
            words,
            1,
            TileSpec::new(Projection::new(
                &[axis, KB],
                &[
                    PhysicalAxisMap::of(axis),
                    PhysicalAxisMap::of(KB).over(STEP / BLOCK),
                ],
            )),
        );
        let global = TileArgLaunch::new(
            tensor(client.create_from_slice(f32::as_bytes(&[global])), 1, 1),
            words,
            1,
            TileSpec::new(Projection::new(
                &[M, N, KB],
                &[PhysicalAxisMap::broadcast()],
            ))
            .packed(Field::Float(FloatKind::F32)),
        );
        values.scales(block_scales).scales(global)
    };
    block_scaled_matmul_under_words::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        operand(
            codes(&a_codes),
            scale_words(&a_scale_values),
            m,
            M,
            a_global,
        ),
        operand(
            codes(&b_codes),
            scale_words(&b_scale_values),
            n,
            N,
            b_global,
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            element,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        element,
    );

    let decoded = |code: u8| e2m1::from_bits(code).to_f32();
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k)
                .map(|l| {
                    decoded(a_codes[i * k + l])
                        * a_scale_values[i * blocks + l / BLOCK]
                        * decoded(b_codes[j * k + l])
                        * b_scale_values[j * blocks + l / BLOCK]
                })
                .sum::<f32>()
                * a_global
                * b_global;
            let have = got.get_f32(&[i, j]);
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "k = {k} in blocks, element ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// Whether `client`'s device offers the block-scaled `16 × 8 × 64` instruction over `e2m1`
/// operands under `e4m3` scales.
pub(crate) fn offers_nvfp4(client: &Client) -> bool {
    let fp4 = ElemType::Float(FloatKind::E2M1x2);
    client
        .properties()
        .features
        .matmul
        .scaled_mma
        .iter()
        .any(|config| {
            config.a_type == fp4
                && config.b_type == fp4
                && config.scales_type == ElemType::Float(FloatKind::E4M3)
                && (config.m, config.n, config.k, config.scales_factor) == (16, 8, 64, 4)
        })
}

/// One instruction's depth: a single stage, a single step.
#[test]
fn one_block_scaled_step_matches_the_reference() {
    check_block_scaled(64, None);
}

/// Several stages of `K`, each a step of the instruction, summed into one accumulator.
#[test]
fn a_walk_of_block_scaled_steps_matches_the_reference() {
    check_block_scaled(256, None);
}

/// NVFP4's second level: a factor over each whole operand, which the instruction holds no level
/// for, multiplied into the sum before it drains.
#[test]
fn per_tensor_factors_scale_the_block_scaled_sum() {
    check_block_scaled(128, Some((0.375, 1.0 / 448.0)));
}

/// The product stated in `k`'s blocks, its block scales packed `e4m3` fields: each instruction
/// step spans four blocks and reads their four scales as one stored word.
#[test]
fn block_scaled_steps_read_packed_scales_in_blocks() {
    check_block_scaled_in_blocks(256, (0.375, 1.0 / 448.0));
}
