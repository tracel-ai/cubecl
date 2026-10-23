//! `acc.mma(a, b)` through the manual-mma instruction at a leaf deeper along `k` than one
//! instruction: the fragments hold `k` in lane runs, each unit's run of the window read in whole
//! served loads, and the contraction issues one instruction after another into the accumulator.
//!
//! Two operands a decode reads that way: a plain 16-bit one, and `e2m1` codes under one level of
//! `e4m3` block scales, decoded and scaled in registers where the instruction reads them. The
//! kernel names neither: the leaf's depth is the whole statement. Out of a stage in shared memory
//! the same fragment loads a step at a time instead, each step in the instruction's own order.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    bytes::Bytes,
    cmma::{MatrixIdent, MmaDefinition},
    ir::{FloatKind, types::Fp8Format},
    prelude::*,
    quant::scheme::QuantValue,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubecl_common::{e2m1, e4m3};
use cubecl_tile::{
    Axis, Instruction, Level, Levels, MmaOperands, Partitioning, Projection, Semiring, Space,
    StageStorage, Stages, TileArg, TileArgLaunch, TileSpec, kind::Field, layout::PhysicalAxisMap,
    ops::matmul::MmaIo,
};
use half::f16;

use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// `k` as the two digits a scale block makes of it: the block, and the value inside it.
const KB: Axis = Axis(3);
const KI: Axis = Axis(4);

/// The leaf's depth along `k`: eight instructions of sixteen, a unit's run thirty-two values.
const WINDOW: usize = 128;
/// Values one scale covers along `k`: NVFP4's block.
const BLOCK: usize = 16;
/// `e4m3` scales one word of them holds.
const SCALES_PER_WORD: usize = 4;

/// What the kernel reads out of a stage in shared memory rather than straight from its tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Staged {
    Nothing,
    /// The activation alone: the weight's fragments still load in lane runs.
    Activation,
    /// Both operands, the weight as it is stored, its scale words beside it: the weight's
    /// fragments load a step at a time.
    Both,
}

/// `c = a · b`, a leaf of `WINDOW` along `k` at a time, through the manual-mma instruction, `b`
/// dequantized at `decoded` into the fragments it meets. An operand under block scales carries
/// them; a per-tensor scale above them leaves `b` for the sum (`Tile::factor`). Each operand read
/// straight, or out of a stage in shared memory as `staged` says.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn lane_run_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] staged: Staged,
    #[comptime] decoded: ElemType,
) {
    let instruction = comptime!(Instruction::Mma {
        io: MmaIo::manual(),
        operands: MmaOperands::Plain,
    });
    let a = a.tile(&space);
    let mut c = c.tile(&space);
    let (b, b_factor) = b.tile(&space).factor(c.space(), instruction);
    let acc = c.accumulator::<f32>(&a, &b, instruction, Semiring::SUM_PROD);
    match comptime!(staged) {
        Staged::Nothing => {
            for region in space.over(&level) {
                let mut acc_r = acc.at(&region);
                let b_r = acc_r.dequantized(&b.at(&region), decoded);
                acc_r.mma(&a.at(&region), &b_r);
            }
        }
        Staged::Activation => {
            let walk = space.over(&level);
            let mut stages = Stages::smem_single(&walk, &a, StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, region| {
                let mut acc_r = acc.at(region);
                let b_r = acc_r.dequantized(&b.at(region), decoded);
                slot.consume(|a_r| {
                    acc_r.mma(a_r, &b_r);
                });
            });
        }
        Staged::Both => {
            let walk = space.over(&level);
            let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, region| {
                let mut acc_r = acc.at(region);
                slot.consume(|a_r, b_r| {
                    let b_r = acc_r.dequantized(b_r, decoded);
                    acc_r.mma(a_r, &b_r);
                });
            });
        }
    }
    c.dequantize_from(&acc.scaled(&b_factor));
}

/// Whether the device offers the 16-bit `16 × 8 × 16` instruction the lane runs are read off.
fn offers_m16n8k16(client: &cubecl::client::Client) -> bool {
    client
        .properties()
        .features
        .matmul
        .mma
        .iter()
        .any(|config| {
            config.a_type == f16::elem_type_native()
                && (config.m, config.n, config.k) == (16, 8, 16)
        })
}

fn skip_without_instruction(client: &cubecl::client::Client) -> bool {
    let offered = offers_m16n8k16(client);
    if !offered {
        ValidationResult::Skipped(
            "the device offers no 16-bit 16x8x16 manual-mma instruction".to_string(),
        )
        .enforce();
    }
    !offered
}

/// The `16 × 8` product's partitioning, `k` deep, walked a `WINDOW` at a time.
fn launcher(client: &cubecl::client::Client, k: usize) -> cubecl_tile::Launcher {
    implied(
        client,
        Partitioning::new(
            Space::new(&[(M, 16), (N, 8), (K, k)]),
            Levels::leaf(&[(M, 16), (N, 8), (K, WINDOW)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    )
}

fn tensor(handle: cubecl::server::Handle, rows: usize, cols: usize) -> TensorArg {
    unsafe { TensorArg::from_raw_parts(handle, [cols, 1].into(), [rows, cols].into()) }
}

/// Values small enough, and on a grid fine enough, that every product and sum is exact in `f32`.
fn activation(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i * 7 + i / k * 3) % 17) as f32 / 8.0 - 1.0)
        .collect()
}

fn assert_product(
    got: &HostData,
    rows: usize,
    n: usize,
    want: impl Fn(usize, usize) -> f32,
    what: &str,
) {
    for i in 0..rows {
        for j in 0..n {
            let (have, want) = (got.get_f32(&[i, j]), want(i, j));
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "{what}, element ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// A plain 16-bit `a · b`, `b` held `[n, k]`, each operand served `width_a` and `width_b` wide.
fn check_plain(k: usize, width_a: usize, width_b: usize) {
    let client = cubecl::test_device().client();
    if skip_without_instruction(&client) {
        return;
    }
    let launcher = launcher(&client, k);
    let a = activation(16, k);
    let b: Vec<f32> = (0..8 * k)
        .map(|i| ((i * 5 + i / k) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let half = |values: &[f32]| values.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
    let a_handle = client.create_from_slice(f16::as_bytes(&half(&a)));
    let b_handle = client.create_from_slice(f16::as_bytes(&half(&b)));
    let c = TestInput::builder(client.clone(), Shape::from(vec![16, 8]))
        .zeros()
        .generate_without_host_data();
    lane_run_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            tensor(a_handle, 16, k),
            f16::elem_type_native(),
            width_a,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            tensor(b_handle, 8, k),
            f16::elem_type_native(),
            width_b,
            TileSpec::direct(&[N, K]),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            c.dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Staged::Nothing,
        f16::elem_type_native(),
    );
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    assert_product(
        &got,
        16,
        8,
        |i, j| (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum(),
        &format!("plain, k = {k}, served {width_a} and {width_b} wide"),
    );
}

/// `a · (b ⊗ s)` with `b` NVFP4: `e2m1` codes eight to a word along `k`, `e4m3` scales four to a
/// word, one every sixteen codes, the codes served `words` words a load. `k` is spelled as the
/// two digits a block makes of it, `(KB, KI)`, so the scales are the operand omitting `KI`.
/// `columns` fragments of eight columns share the plane, each contracting the one `a` it loads.
fn check_nvfp4(k: usize, width_a: usize, words: usize, columns: usize) {
    check_nvfp4_rows(16, k, width_a, words, columns, Staged::Nothing, None)
}

/// [`check_nvfp4`] at `rows` rows of the fragment's sixteen: a product of fewer rows overhangs
/// the fragment, its rows past the edge read as zeros and drained nowhere. Operands read out of a
/// stage in shared memory as `staged` says, a stage holding the product's rows alone. A staged
/// weight's scales are words of four, which its stage keeps beside the codes. Where `global`
/// is stated, the weight carries it as a per-tensor scale above its block scales.
fn check_nvfp4_rows(
    rows: usize,
    k: usize,
    width_a: usize,
    words: usize,
    columns: usize,
    staged: Staged,
    global: Option<f32>,
) {
    let client = cubecl::test_device().client();
    if skip_without_instruction(&client) {
        return;
    }
    let blocks = k / BLOCK;
    let n = 8 * columns;
    let leaf = Levels::leaf(&[(M, 16), (N, 8), (KB, WINDOW / BLOCK), (KI, BLOCK)]).walk_every(&[M]);
    // More than one column of fragments: a level below the walk steps them, `k` held whole.
    let leaf = match columns {
        1 => leaf,
        _ => leaf.walk(&[(N, columns)]),
    };
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, n), (KB, blocks), (KI, BLOCK)]),
            leaf.walk_every(&[N, KB, KI]).build(),
        ),
        Form::Static,
    );
    let a = activation(rows, k);
    // Every code, varied by position so no column repeats another.
    let codes: Vec<u8> = (0..n * k)
        .map(|i| ((i * 5 + 3 * (i / k)) % 16) as u8)
        .collect();
    // Scales exact in `e4m3`: a power of two times one of its mantissas.
    let scale_values: Vec<f32> = (0..n * blocks)
        .map(|i| (1 + i % 8) as f32 * 2f32.powi((i % 5) as i32 - 3))
        .collect();
    let pack = |bytes: Vec<u8>, per_word: usize| -> Vec<u32> {
        let bits = 32 / per_word;
        bytes
            .chunks(per_word)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &b)| acc | (b as u32) << (bits * j))
            })
            .collect()
    };
    let scale_bytes: Vec<u8> = scale_values
        .iter()
        .map(|&s| e4m3::from_f32(s).to_bits())
        .collect();
    let half: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
    let tensor = |bytes: Bytes, shape: Vec<usize>, dtype: ElemType| {
        TensorHandle::new_contiguous(shape, client.create(bytes), dtype).binding()
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
    let a_op = launcher
        .arg(
            tensor(
                Bytes::from_elems(half),
                vec![rows, blocks, BLOCK],
                f16::elem_type_native(),
            ),
            f16::elem_type_native(),
        )
        // A dim a digit, as a product restates its activation: every axis an identity, so a
        // fragment's rows past the product's are checked at the edge.
        .axes(&[M, KB, KI])
        .vectorize(width_a)
        .build()
        .unwrap();
    let b_op = launcher
        .arg(
            tensor(
                Bytes::from_elems(pack(codes.clone(), 8)),
                vec![n, k],
                u32::elem_type_native(),
            ),
            u32::elem_type_native(),
        )
        .gathered(along_k(N))
        .packed(Field::Quant(QuantValue::E2M1).read_on(&client))
        .vectorize(words * 8)
        .build()
        .unwrap();
    // One scale per `(column, block of K)`: `KI` is carried and addressed by nothing. A staged
    // weight's scales stay the words of four they are stored as; otherwise they are read as
    // numbers.
    let scale_words = Bytes::from_elems(pack(scale_bytes, SCALES_PER_WORD));
    let s_op = match staged {
        Staged::Both => launcher
            .arg(
                tensor(
                    scale_words,
                    vec![n, blocks / SCALES_PER_WORD],
                    u32::elem_type_native(),
                ),
                u32::elem_type_native(),
            )
            .gathered(Projection::new(
                &[N, KB],
                &[
                    PhysicalAxisMap::of(N),
                    PhysicalAxisMap::of(KB).over(SCALES_PER_WORD),
                ],
            ))
            .build(),
        Staged::Nothing | Staged::Activation => launcher
            .arg(
                tensor(scale_words, vec![n, blocks], u32::elem_type_native()),
                u32::elem_type_native(),
            )
            .axes(&[N, KB])
            .packed(Field::Fp8(Fp8Format::E4M3))
            .vectorize(SCALES_PER_WORD)
            .build(),
    }
    .unwrap();
    let c = TestInput::builder(client.clone(), Shape::from(vec![rows, n]))
        .zeros()
        .generate_without_host_data();
    let c_op = launcher
        .arg(c.clone().binding(), c.dtype)
        .axes(&[M, N])
        .build()
        .unwrap();
    lane_run_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        match global {
            None => b_op.arg().scales(s_op.arg()),
            Some(value) => {
                // One `f32` over the whole weight, read as the word it is stored in.
                let global = launcher
                    .arg(
                        tensor(
                            Bytes::from_elems(vec![value.to_bits()]),
                            vec![1],
                            u32::elem_type_native(),
                        ),
                        u32::elem_type_native(),
                    )
                    .gathered(Projection::new(
                        &[N, KB, KI],
                        &[PhysicalAxisMap::broadcast()],
                    ))
                    .packed(Field::Float(FloatKind::F32))
                    .build()
                    .unwrap();
                b_op.arg().scales(s_op.arg()).scales(global.arg())
            }
        },
        c_op.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        staged,
        f16::elem_type_native(),
    );
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    let decoded = |code: u8| e2m1::from_bits(code).to_f32();
    assert_product(
        &got,
        rows,
        n,
        |i, j| {
            (0..k)
                .map(|l| {
                    a[i * k + l] * decoded(codes[j * k + l]) * scale_values[(j * k + l) / BLOCK]
                })
                .sum::<f32>()
                * global.unwrap_or(1.0)
        },
        &format!(
            "nvfp4, {rows} rows, k = {k}, a served {width_a} wide{}, codes {words} words a \
             load, {columns} column fragments{}",
            match staged {
                Staged::Nothing => "",
                Staged::Activation => " out of a stage",
                Staged::Both => ", both out of stages",
            },
            match global {
                None => "",
                Some(_) => ", under a per-tensor scale",
            }
        ),
    );
}

/// [`check_plain`] with `k` spelled as `(KB, KI)` on both operands, as a block-scaled operand's
/// partner is.
fn check_plain_split(k: usize, width: usize) {
    let client = cubecl::test_device().client();
    if skip_without_instruction(&client) {
        return;
    }
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 16), (N, 8), (KB, k / BLOCK), (KI, BLOCK)]),
            Levels::leaf(&[(M, 16), (N, 8), (KB, WINDOW / BLOCK), (KI, BLOCK)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );
    let a = activation(16, k);
    let b: Vec<f32> = (0..8 * k)
        .map(|i| ((i * 5 + i / k) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let half = |values: &[f32]| values.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
    let tensor = |values: &[f32], rows: usize| {
        TensorHandle::new_contiguous(
            vec![rows, k],
            client.create(Bytes::from_elems(half(values))),
            f16::elem_type_native(),
        )
        .binding()
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
    let a_op = launcher
        .arg(tensor(&a, 16), f16::elem_type_native())
        .gathered(along_k(M))
        .vectorize(width)
        .build()
        .unwrap();
    let b_op = launcher
        .arg(tensor(&b, 8), f16::elem_type_native())
        .gathered(along_k(N))
        .vectorize(width)
        .build()
        .unwrap();
    let c = TestInput::builder(client.clone(), Shape::from(vec![16, 8]))
        .zeros()
        .generate_without_host_data();
    let c_op = launcher
        .arg(c.clone().binding(), c.dtype)
        .axes(&[M, N])
        .build()
        .unwrap();
    lane_run_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        c_op.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Staged::Nothing,
        f16::elem_type_native(),
    );
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    assert_product(
        &got,
        16,
        8,
        |i, j| (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum(),
        &format!("plain over (KB, KI), k = {k}, served {width} wide"),
    );
}

/// Plain values over `k` spelled as two digits.
#[test]
fn a_plain_window_over_two_digits_of_k_contracts_in_lane_runs() {
    check_plain_split(WINDOW, 8);
    check_plain_split(2 * WINDOW, 1);
}

/// One window of plain values, read a whole 16-byte line at a time.
#[test]
fn a_plain_window_contracts_in_lane_runs() {
    check_plain(WINDOW, 8, 8);
}

/// Several windows walked into one accumulator, and lines narrower than a run.
#[test]
fn a_walk_of_plain_windows_contracts_in_lane_runs() {
    check_plain(4 * WINDOW, 8, 8);
    check_plain(2 * WINDOW, 2, 4);
    check_plain(WINDOW, 1, 1);
}

/// NVFP4 codes read a block, two words, a load, decoded and scaled in registers: in rows, a
/// line holds no more than one block, since a block is a digit of `k` of its own.
#[test]
fn an_nvfp4_window_contracts_in_lane_runs() {
    check_nvfp4(WINDOW, 8, 2, 1);
}

/// Several NVFP4 windows, and codes read a word a load.
#[test]
fn a_walk_of_nvfp4_windows_contracts_in_lane_runs() {
    check_nvfp4(3 * WINDOW, 8, 2, 1);
    check_nvfp4(WINDOW, 4, 1, 1);
    check_nvfp4(2 * WINDOW, 1, 1, 1);
}

/// Two and four fragments of columns on one plane, each contracting the one `a` the plane loads
/// for all of them.
#[test]
fn column_fragments_share_one_lane_run_of_a() {
    check_nvfp4(2 * WINDOW, 8, 2, 2);
    check_nvfp4(WINDOW, 8, 2, 4);
}

/// Fewer rows than a fragment's, one fragment and a grid of them: the rows past the product's
/// read as zeros and drain nowhere.
#[test]
fn a_product_of_fewer_rows_overhangs_its_fragments() {
    check_nvfp4_rows(3, 2 * WINDOW, 8, 2, 1, Staged::Nothing, None);
    check_nvfp4_rows(3, 2 * WINDOW, 8, 2, 2, Staged::Nothing, None);
    check_nvfp4_rows(9, WINDOW, 8, 2, 4, Staged::Nothing, None);
}

/// The activation out of a stage in shared memory, which holds the product's rows alone: the
/// units covering rows past them read zeros rather than past the stage.
#[test]
fn a_staged_activation_of_fewer_rows_than_a_fragment_contracts_in_lane_runs() {
    check_nvfp4_rows(1, 2 * WINDOW, 8, 2, 1, Staged::Activation, None);
    check_nvfp4_rows(3, 2 * WINDOW, 8, 2, 2, Staged::Activation, None);
    check_nvfp4_rows(9, WINDOW, 8, 2, 1, Staged::Activation, None);
    check_nvfp4_rows(16, WINDOW, 8, 2, 1, Staged::Activation, None);
}

/// **A weight staged as it is stored decodes at the kernel's element a step at a time.** Out of
/// shared memory the weight's fragment, dequantized at `f16`, loads each instruction step in the
/// instruction's own order, decoding its codes under their scale words as it loads.
#[test]
fn a_staged_nvfp4_weight_dequantizes_a_step_at_a_time() {
    check_nvfp4_rows(16, WINDOW, 8, 2, 1, Staged::Both, None);
    check_nvfp4_rows(16, 3 * WINDOW, 8, 2, 1, Staged::Both, None);
    check_nvfp4_rows(3, 2 * WINDOW, 8, 2, 2, Staged::Both, None);
}

/// **A per-tensor scale above the block scales leaves the weight for the sum.** Lane runs convert
/// codes under one level of scales; the per-tensor scale above them is taken out by
/// `Tile::factor` and multiplies the sum once, the weight read straight or out of a stage.
#[test]
fn a_per_tensor_scale_above_lane_runs_scales_the_sum() {
    check_nvfp4_rows(16, 2 * WINDOW, 8, 2, 1, Staged::Nothing, Some(0.375));
    check_nvfp4_rows(3, WINDOW, 8, 2, 2, Staged::Nothing, Some(1.5));
    check_nvfp4_rows(16, WINDOW, 8, 2, 1, Staged::Both, Some(0.375));
}

/// Each unit's register `i` of instruction role `ident` holds the cells the lane-run map says it
/// does, as `position_of_nth` places them: the run of the unit's quad along `k`, and the rows (an
/// `A`) or the column (a `B`) the unit holds. Written to `out` as `(row, k)` pairs per cell.
#[cube(launch)]
fn positions(out: &mut [u32], #[comptime] ident: MatrixIdent) {
    let def = MmaDefinition::<f16, f16, f32>::new(16usize, 8usize, 16usize);
    let cells = def.elems_per_lane(ident);
    let unit = UNIT_POS_PLANE;
    #[unroll]
    for e in 0..cells {
        let (row, col) = def.position_of_nth(unit, e as u32, ident);
        let at = ((unit as usize) * cells + e) * 2;
        out[at] = row;
        out[at + 1] = col;
    }
}

/// The layout the lane-run map is written for, against the device's own: unit `u` holds quad
/// `u % 4` of `k`, cells `2q + 8p + e` for pair `p` and cell `e` of a register; an `A`'s register
/// `i` row `u / 4 + 8 (i % 2)` and pair `i / 2`, a `B`'s pair `i` and column `u / 4`.
#[test]
fn the_lane_run_map_is_the_devices_layout() {
    let client = cubecl::test_device().client();
    if skip_without_instruction(&client) {
        return;
    }
    for (ident, cells) in [(MatrixIdent::A, 8usize), (MatrixIdent::B, 4usize)] {
        let out = client.empty(32 * cells * 2 * 4);
        unsafe {
            positions::launch(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(out.clone(), 32 * cells * 2),
                ident,
            )
        };
        let bytes = client.read_one(out).unwrap();
        let got = u32::from_bytes(&bytes);
        for unit in 0..32u32 {
            let (g, q) = (unit / 4, unit % 4);
            for e in 0..cells {
                let (register, cell) = (e / 2, (e % 2) as u32);
                let at = (unit as usize * cells + e) * 2;
                let (row, col) = (got[at], got[at + 1]);
                let (want_row, want_col) = match ident {
                    MatrixIdent::A => (
                        g + 8 * (register as u32 % 2),
                        2 * q + 8 * (register as u32 / 2) + cell,
                    ),
                    _ => (2 * q + 8 * register as u32 + cell, g),
                };
                assert_eq!(
                    (row, col),
                    (want_row, want_col),
                    "{ident:?}, unit {unit}, cell {e}"
                );
            }
        }
    }
}
