//! What the block-scaled instruction delivers through the tile API when loading is out of the
//! way: each cube is one plane holding a `64 × 64` grid of fragments, its operands staged in
//! shared memory once and contracted `repeats` times over the same stage, so a fragment load is
//! shared by the whole grid and global memory is read once. The same kernel at `f16`, through
//! the plain manual-mma instruction, is the yardstick.
//!
//! Two forms of scales: one `f32` per block, read and narrowed one at a time, and words of four
//! `e4m3` scales, as an NVFP4 checkpoint packs them, each one load straight into the
//! instruction's scale register.
//!
//! The correctness checks run one cube against the host sum. The throughput comparison is a
//! measurement, ignored by default:
//!
//! ```sh
//! cargo test -p cubecl-tile --release --features cubecl/cuda,cubecl/cuda-cpp --test lib \
//!     -- block_scaled_peak --ignored --nocapture
//! ```

use std::time::Instant;

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{
    client::Client, ir::FloatKind, ir::types::Fp8Format, prelude::*, quant::scheme::QuantValue,
    zspace::Shape,
};
use cubecl_common::{e2m1, e4m3};
use cubecl_tile::{
    Axis, BlockScaledOperands, Instruction, Level, Levels, MmaOperands, Partitioning, Projection,
    Semiring, Space, StageStorage, Stages, TileArg, TileArgLaunch, TileSpec, kind::Field,
    layout::PhysicalAxisMap, ops::matmul::MmaIo,
};

use super::{Form, implied};

/// NVFP4 as the block-scaled instruction takes it: `e2m1` codes two to a byte under an `e4m3`
/// scale every sixteen of `k`.
const NVFP4: BlockScaledOperands = BlockScaledOperands {
    lhs: FloatKind::E2M1x2,
    rhs: FloatKind::E2M1x2,
    scales: FloatKind::E4M3,
    block: 16,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Values one scale covers along `k`: NVFP4's block.
const BLOCK: usize = 16;

/// Values of `k` one instruction step covers, and so one word of scales.
const STEP: usize = 64;

/// A cube's box the correctness checks run: one plane's accumulator, and one stage's depth.
const BOX: (usize, usize, usize) = (64, 64, 64);

/// The depth of one stage.
const DEPTH: usize = 64;

/// How the block scales are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaleForm {
    /// One `f32` per block.
    Floats,
    /// One `u32` per instruction step: its four blocks' `e4m3` scales, the first in the low byte.
    Words,
}

/// `c = repeats · (a ⊗ sa) · (b ⊗ sb)` per cube, over one stage contracted `repeats` times.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn block_scaled_peak<E: Numeric>(
    a: &TileArg<'_>,
    a_scales: &TileArg<'_>,
    b: &TileArg<'_>,
    b_scales: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] repeats: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in space.over(&cubes) {
        let (a, b, c) = (a.at(&cube), b.at(&cube), c.at(&cube));
        let acc = c.accumulator::<E>(
            &a,
            &b,
            comptime!(Instruction::Mma {
                io: MmaIo::manual(),
                operands: MmaOperands::BlockScaled(NVFP4),
            }),
            Semiring::SUM_PROD,
        );
        let walk = cube.over(&steps);
        let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);
        stages.pipelined(walk, |slot, region| {
            let acc_r = acc.at(region);
            slot.consume(|a_s, b_s| {
                let a_s = a_s.mul(&a_scales.tile(&space).at(region));
                let b_s = b_s.mul(&b_scales.tile(&space).at(region));
                for plane in region {
                    let acc_p = acc_r.at(&plane);
                    let (a_p, b_p) = (a_s.at(&plane), b_s.at(&plane));
                    for _ in 0..repeats {
                        for step in plane.walk().unrolled() {
                            let mut acc_s = acc_p.at(&step);
                            acc_s.mma(&a_p.at(&step), &b_p.at(&step));
                        }
                    }
                }
            });
        });
        acc.drained_into(&c);
    }
}

/// `c = repeats · a · bᵀ` per cube at `f16`, the same walk through the plain manual-mma
/// instruction.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn float_peak(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] repeats: usize,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in space.over(&cubes) {
        let (a, b, c) = (a.at(&cube), b.at(&cube), c.at(&cube));
        let acc = c.accumulator::<f32>(
            &a,
            &b,
            comptime!(Instruction::Mma {
                io: MmaIo::manual(),
                operands: MmaOperands::Plain,
            }),
            Semiring::SUM_PROD,
        );
        let walk = cube.over(&steps);
        let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);
        stages.pipelined(walk, |slot, region| {
            let acc_r = acc.at(region);
            slot.consume(|a_s, b_s| {
                for plane in region {
                    let acc_p = acc_r.at(&plane);
                    let (a_p, b_p) = (a_s.at(&plane), b_s.at(&plane));
                    for _ in 0..repeats {
                        for step in plane.walk().unrolled() {
                            let mut acc_s = acc_p.at(&step);
                            acc_s.mma(&a_p.at(&step), &b_p.at(&step));
                        }
                    }
                }
            });
        });
        acc.drained_into(&c);
    }
}

/// The planes of a cube and the box each holds.
#[derive(Clone, Copy, Debug)]
struct CubeShape {
    /// Planes along `m` and `n`.
    planes: (usize, usize),
    /// One plane's box along `m` and `n`.
    plane: (usize, usize),
}

impl CubeShape {
    /// The cube's box along `m` and `n`.
    fn cube(self) -> (usize, usize) {
        (self.planes.0 * self.plane.0, self.planes.1 * self.plane.1)
    }
}

/// The partitioning of a `cubes_m × cubes_n` grid of cubes shaped as `shape`, one stage deep,
/// leaf up: one instruction `instruction_k` deep, a plane's grid of them over its box, the steps
/// of `K` through the stage, the cube's planes, the stage, and the cubes.
fn partitioning(
    (cubes_m, cubes_n): (usize, usize),
    shape: CubeShape,
    instruction_k: usize,
) -> Partitioning {
    let bk = DEPTH;
    let (im, in_) = (16, 8);
    let (pm, pn) = shape.plane;
    let (cm, cn) = shape.cube();
    Partitioning::new(
        Space::new(&[(M, cm * cubes_m), (N, cn * cubes_n), (K, bk)]),
        Levels::leaf(&[(M, im), (N, in_), (K, instruction_k)])
            .walk(&[(M, pm / im), (N, pn / in_)])
            .walk(&[(K, bk / instruction_k)])
            .planes(&[(M, shape.planes.0), (N, shape.planes.1)])
            .walk_every(&[K])
            .cubes(&[M, N])
            .build(),
    )
}

/// A `rows × cols` row-major buffer.
fn tensor(handle: cubecl::server::Handle, rows: usize, cols: usize) -> TensorArg {
    unsafe { TensorArg::from_raw_parts(handle, [cols, 1].into(), [rows, cols].into()) }
}

/// NVFP4 operands: codes and block scales, row-major along `k`.
struct Nvfp4Operands {
    a_codes: Vec<u8>,
    b_codes: Vec<u8>,
    a_scales: Vec<f32>,
    b_scales: Vec<f32>,
}

impl Nvfp4Operands {
    fn new(m: usize, n: usize, k: usize) -> Self {
        // Scales exact in `e4m3`: a power of two times one of its mantissas.
        let scale = |i: usize| (1 + i % 8) as f32 * 2f32.powi((i % 5) as i32 - 3);
        Self {
            a_codes: (0..m * k).map(|i| ((i * 7 + i / k) % 16) as u8).collect(),
            b_codes: (0..n * k)
                .map(|i| ((i * 5 + 3 * (i / k)) % 16) as u8)
                .collect(),
            a_scales: (0..m * k / BLOCK).map(scale).collect(),
            b_scales: (0..n * k / BLOCK).map(|i| scale(i + 3)).collect(),
        }
    }

    /// Eight codes to a word, the first in the low bits.
    fn code_words(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks(8)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &code)| acc | (code as u32) << (4 * j))
            })
            .collect()
    }

    /// Four `e4m3` scales to a word, the first in the low byte.
    fn scale_words(scales: &[f32]) -> Vec<u32> {
        scales
            .chunks(STEP / BLOCK)
            .map(|word| {
                word.iter().enumerate().fold(0u32, |acc, (j, &scale)| {
                    acc | (e4m3::from_f32(scale).to_bits() as u32) << (8 * j)
                })
            })
            .collect()
    }
}

/// Launch the NVFP4 kernel over a `cubes_m × cubes_n` grid, `repeats` contractions per cube, its
/// scales stored as `form` says, and hand back the output.
fn launch_nvfp4(
    client: &Client,
    operands: &Nvfp4Operands,
    (cubes_m, cubes_n): (usize, usize),
    shape: CubeShape,
    repeats: usize,
    form: ScaleForm,
) -> cubecl::std::tensor::TensorHandle {
    let (bm, bn) = shape.cube();
    let (m, n, k) = (bm * cubes_m, bn * cubes_n, DEPTH);
    let launcher = implied(
        client,
        partitioning((cubes_m, cubes_n), shape, STEP),
        Form::Static,
    );
    let c = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();
    let values = |rows: Axis| {
        TileSpec::new(Projection::direct(&[rows, K])).packed(Field::Quant(QuantValue::E2M1))
    };
    let per = match form {
        ScaleForm::Floats => BLOCK,
        ScaleForm::Words => STEP,
    };
    // Floats are bound as the words they are stored in, and served as `f32`; words as they are.
    let scales = |rows: Axis| {
        let spec = TileSpec::new(Projection::new(
            &[rows, K],
            &[PhysicalAxisMap::of(rows), PhysicalAxisMap::of(K).over(per)],
        ));
        match form {
            ScaleForm::Floats => spec.packed(Field::Float(FloatKind::F32)),
            ScaleForm::Words => spec.words(Field::Fp8(Fp8Format::E4M3)),
        }
    };
    let upload_scales = |scales: &[f32], rows: usize| {
        let handle = match form {
            ScaleForm::Floats => client.create_from_slice(f32::as_bytes(scales)),
            ScaleForm::Words => {
                client.create_from_slice(u32::as_bytes(&Nvfp4Operands::scale_words(scales)))
            }
        };
        tensor(handle, rows, k / per)
    };
    let upload_codes = |codes: &[u8], rows: usize| {
        tensor(
            client.create_from_slice(u32::as_bytes(&Nvfp4Operands::code_words(codes))),
            rows,
            k,
        )
    };
    // Values and scales are bound at the words their buffers hold.
    let words = u32::elem_type_native();
    let element = f32::elem_type_native();
    block_scaled_peak::launch(
        client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(upload_codes(&operands.a_codes, m), words, 1, values(M)),
        TileArgLaunch::new(upload_scales(&operands.a_scales, m), words, 1, scales(M)),
        TileArgLaunch::new(upload_codes(&operands.b_codes, n), words, 1, values(N)),
        TileArgLaunch::new(upload_scales(&operands.b_scales, n), words, 1, scales(N)),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            element,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        repeats,
        element,
    );
    c
}

/// Launch the `f16` kernel over a `cubes_m × cubes_n` grid of zeros, `repeats` contractions per
/// cube.
fn launch_f16(
    client: &Client,
    (cubes_m, cubes_n): (usize, usize),
    shape: CubeShape,
    repeats: usize,
) {
    let (bm, bn) = shape.cube();
    let (m, n, k) = (bm * cubes_m, bn * cubes_n, DEPTH);
    let launcher = implied(
        client,
        partitioning((cubes_m, cubes_n), shape, 16),
        Form::Static,
    );
    let zeros = |rows: usize| client.create_from_slice(&vec![0u8; rows * k * 2]);
    let c = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();
    let rows = |axis: Axis| TileSpec::new(Projection::direct(&[axis, K]));
    let element = half::f16::elem_type_native();
    float_peak::launch(
        client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(tensor(zeros(m), m, k), element, 1, rows(M)),
        TileArgLaunch::new(tensor(zeros(n), n, k), element, 1, rows(N)),
        TileArgLaunch::new(
            c.binding().into_tensor_arg(),
            f32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        repeats,
    );
}

/// One cube, three contractions of its stage, its scales stored as `form` says: `c` is three
/// times the host sum of the decoded operands under their scales.
fn check_repeated_stage(form: ScaleForm) {
    let client = cubecl::test_device().client();
    if !super::block_scaled::offers_nvfp4(&client) {
        return;
    }
    let (m, n, k) = BOX;
    let repeats = 3;
    let operands = Nvfp4Operands::new(m, n, k);
    let shape = CubeShape {
        planes: (2, 2),
        plane: (m / 2, n / 2),
    };
    let c = launch_nvfp4(&client, &operands, (1, 1), shape, repeats, form);
    let decoded = |code: u8| e2m1::from_bits(code).to_f32();
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k)
                .map(|l| {
                    decoded(operands.a_codes[i * k + l])
                        * operands.a_scales[i * (k / BLOCK) + l / BLOCK]
                        * decoded(operands.b_codes[j * k + l])
                        * operands.b_scales[j * (k / BLOCK) + l / BLOCK]
                })
                .sum::<f32>()
                * repeats as f32;
            let have = got.get_f32(&[i, j]);
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "{form:?}, element ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

#[test]
fn a_repeated_stage_under_float_scales_matches_the_reference() {
    check_repeated_stage(ScaleForm::Floats);
}

#[test]
fn a_repeated_stage_under_words_of_scales_matches_the_reference() {
    check_repeated_stage(ScaleForm::Words);
}

/// The kernels' rates over a grid that fills the device many times over, each the best of a few
/// timed launches after a warm one.
#[test]
#[ignore = "a measurement: run it deliberately, in release"]
fn block_scaled_peak_against_f16() {
    let client = cubecl::test_device().client();
    if !super::block_scaled::offers_nvfp4(&client) {
        eprintln!("the device offers no block-scaled e2m1 instruction");
        return;
    }
    // A 2048 × 2048 output whatever the box, contracted long enough a launch, a few ms, for the
    // clocks to have ramped by the timed ones.
    let side = 2048;
    let repeats = 4096;
    let flops = 2.0 * (side * side * DEPTH) as f64 * repeats as f64;
    let operands = Nvfp4Operands::new(side, side, DEPTH);
    let time = |launch: &dyn Fn()| {
        for _ in 0..3 {
            launch();
        }
        cubecl::future::block_on(client.sync()).unwrap();
        (0..5)
            .map(|_| {
                let start = Instant::now();
                launch();
                cubecl::future::block_on(client.sync()).unwrap();
                start.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min)
    };
    let rate = |seconds: f64| flops / seconds / 1e12;
    let shapes = [
        ((1, 1), (32, 32)),
        ((2, 2), (32, 32)),
        ((4, 2), (32, 32)),
        ((2, 4), (32, 32)),
        ((2, 2), (64, 32)),
        ((4, 2), (16, 64)),
    ];
    for (planes, plane) in shapes {
        let shape = CubeShape { planes, plane };
        let (cm, cn) = shape.cube();
        let grid = (side / cm, side / cn);
        let label = format!(
            "{}x{} planes of {}x{}",
            planes.0, planes.1, plane.0, plane.1
        );
        let f16 = time(&|| launch_f16(&client, grid, shape, repeats));
        eprintln!(
            "PEAK {label:<22} f16            {:8.1} us  {:7.1} TFLOP/s",
            f16 * 1e6,
            rate(f16)
        );
        for form in [ScaleForm::Floats, ScaleForm::Words] {
            let nvfp4 = time(&|| {
                launch_nvfp4(&client, &operands, grid, shape, repeats, form);
            });
            eprintln!(
                "PEAK {label:<22} nvfp4 {:<8} {:8.1} us  {:7.1} TFLOP/s  ({:.2}x f16)",
                format!("{form:?}"),
                nvfp4 * 1e6,
                rate(nvfp4),
                f16 / nvfp4
            );
        }
    }
}
