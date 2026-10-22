//! A contraction cut across a cube's planes, summed back together in shared memory.
//!
//! `split_contraction`'s in-kernel combine one scope down: every plane of a cube contracts its own slice of
//! `K` in registers and drains it into the cube's [`SmemAccumulation`], the cube synchronizes, and
//! the sum leaves through a plain store. Nothing reaches global memory but the finished box, so
//! the output is not zeroed first and binds as an ordinary tile. The planes meet atomically
//! ([`SmemAccumulation`]) or in cyclic rounds ([`SmemCyclicAccumulation`]).
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};

use super::{Form, implied};
use cubecl_tile::ops::matmul::MmaIo;
use cubecl_tile::stage::RowChunks;
use cubecl_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// One box per cube, its `K` distributed to the cube's planes, each plane's slice added into shared
/// memory and the sum stored once.
#[cube(launch)]
fn smem_split_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in &space {
        let mut c_cube = c.at(&cube);
        let sum = c_cube.smem_accumulation::<E>();
        for plane in cube {
            let a_plane = a.at(&plane);
            let b_plane = b.at(&plane);
            let sink_plane = sum.sink.at(&plane);
            let mut partial = sink_plane.accumulator::<E>(
                &a_plane,
                &b_plane,
                comptime!(Instruction::Registers {
                    config: REGISTER_BLOCK
                }),
                Semiring::SUM_PROD,
            );
            partial.mma(&a_plane, &b_plane);
            partial.drained_into(&sink_plane);
        }
        sync_cube();
        c_cube.copy_from(&sum.source);
    }
}

/// The same product, the planes meeting in cyclic rounds rather than atomically: each drains its
/// slice every round, landing one chunk a round, the cube synchronized between rounds.
#[cube(launch)]
fn smem_cyclic_split_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] planes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in &space {
        let mut c_cube = c.at(&cube);
        let sum = c_cube.smem_cyclic_accumulation::<E>(planes);
        for plane in cube {
            let a_plane = a.at(&plane);
            let b_plane = b.at(&plane);
            let mut partial = c_cube.at(&plane).accumulator::<E>(
                &a_plane,
                &b_plane,
                comptime!(Instruction::Registers {
                    config: REGISTER_BLOCK
                }),
                Semiring::SUM_PROD,
            );
            partial.mma(&a_plane, &b_plane);
            sum.drain_at(&partial, &plane, plane.coord(K));
        }
        c_cube.copy_from(&sum.source);
    }
}

fn reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| a(i, p) * b(p, j)).sum()
        })
        .collect()
}

/// How the planes of a cube meet: atomically, or in cyclic rounds into an output served
/// `out_width` scalars a line, which the sum is read back into one scalar at a time. The rhs is
/// served at the output's width, as a register block contracting into it reads it.
#[derive(Clone, Copy)]
enum Meet {
    Atomic,
    Cyclic { out_width: usize },
}

/// `a·b` over boxes of `box_m × box_n`, each box's `K` distributed to `planes` planes of its cube,
/// which meet as `meet` says.
fn run(
    m: usize,
    n: usize,
    k: usize,
    (box_m, box_n): (usize, usize),
    planes: usize,
    meet: Meet,
) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, box_m), (N, box_n), (K, k / planes)])
                .planes(&[(K, planes)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );

    match meet {
        Meet::Atomic => smem_split_matmul::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[M, K]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[K, N]),
            ),
            TileArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[M, N]),
            ),
            launcher.partitioning_arg(),
            dtype,
        ),
        Meet::Cyclic { out_width } => {
            let wide = |handle: &cubecl::std::tensor::TensorHandle, axes: &[Axis]| {
                launcher
                    .arg(handle.clone().binding(), handle.dtype)
                    .axes(axes)
                    .vectorize(out_width)
                    .build()
                    .unwrap()
            };
            let (b, c) = (wide(&b_handle, &[K, N]), wide(&out, &[M, N]));
            smem_cyclic_split_matmul::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                TileArgLaunch::new(
                    a_handle.clone().binding().into_tensor_arg(),
                    dtype,
                    1,
                    TileSpec::direct(&[M, K]),
                ),
                b.arg(),
                c.arg(),
                launcher.partitioning_arg(),
                planes,
                dtype,
            )
        }
    }

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// Whether the device adds `f32` into shared memory atomically; reported rather than silently
/// passed where it cannot.
///
/// The runtime reports the add for buffers, and shared memory is a separate capability it does
/// not report: Vulkan states `shaderSharedFloat32AtomicAdd` apart from the buffer add, and MSL has
/// no threadgroup float atomic at all. So the buffer add is narrowed to the runtimes whose shared
/// memory is known to take it.
fn adds_atomically_in_shared_memory(client: &cubecl::client::Client) -> bool {
    let adds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
        && matches!(client.name(), "hip" | "cuda" | "cpu");
    if !adds {
        ValidationResult::Skipped("device has no f32 atomic add in shared memory".to_string())
            .enforce();
    }
    adds
}

fn assert_matches(got: &HostData, m: usize, n: usize, k: usize) {
    let want = reference(m, n, k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// One cube, four planes: every cell is the sum of four partials that only ever met in shared
/// memory.
#[test]
fn planes_sum_their_slices_in_shared_memory() {
    if !adds_atomically_in_shared_memory(&cubecl::test_device().client()) {
        return;
    }
    let (m, n, k) = (4, 4, 32);
    assert_matches(&run(m, n, k, (4, 4), 4, Meet::Atomic), m, n, k);
}

/// Several cubes, each with its own accumulator: a box's sum never sees another box's planes.
#[test]
fn each_cube_sums_only_its_own_box() {
    if !adds_atomically_in_shared_memory(&cubecl::test_device().client()) {
        return;
    }
    let (m, n, k) = (8, 12, 24);
    assert_matches(&run(m, n, k, (4, 4), 3, Meet::Atomic), m, n, k);
}

/// Four planes meeting in cyclic rounds: every cell is the sum of four partials, each landed by
/// one writer alone in its round, with no atomic, so this runs wherever shared memory does.
#[test]
fn planes_meet_in_cyclic_rounds() {
    let (m, n, k) = (4, 4, 32);
    assert_matches(
        &run(m, n, k, (4, 4), 4, Meet::Cyclic { out_width: 1 }),
        m,
        n,
        k,
    );
}

/// Several cubes, three planes each, meeting in cyclic rounds: three chunks of a box that does not
/// divide by them, and no box sees another's planes.
#[test]
fn each_cube_meets_its_planes_in_cyclic_rounds() {
    let (m, n, k) = (8, 12, 24);
    assert_matches(
        &run(m, n, k, (4, 4), 3, Meet::Cyclic { out_width: 1 }),
        m,
        n,
        k,
    );
}

/// The sum read back one scalar at a time into an output served four scalars a line: each line
/// gathered from four of the shared buffer's cells.
#[test]
fn a_cyclic_sum_lands_in_a_wide_output() {
    let (m, n, k) = (8, 12, 24);
    assert_matches(
        &run(m, n, k, (4, 4), 3, Meet::Cyclic { out_width: 4 }),
        m,
        n,
        k,
    );
}

/// The cyclic rounds fix the order every cell takes its partials in, so two launches sum the same
/// bits, where an atomic meeting may not.
#[test]
fn cyclic_rounds_sum_the_same_every_run() {
    let (m, n, k) = (8, 12, 96);
    let bits = |data: &HostData| -> Vec<u32> {
        (0..m * n)
            .map(|i| data.get_f32(&[i / n, i % n]).to_bits())
            .collect()
    };
    let first = bits(&run(m, n, k, (4, 4), 4, Meet::Cyclic { out_width: 1 }));
    assert_eq!(
        bits(&run(m, n, k, (4, 4), 4, Meet::Cyclic { out_width: 1 })),
        first
    );
}

// -- A fragment written into a window the edge cuts short --------------------------------------
//
// A fragment's store intrinsic writes its whole window, so a tile the problem's edge cuts short
// bounces through the plane's scratch and each unit writes only the cells inside.

/// `c = a · b` in `16×16×16` fragments over boxes that overhang the output, drained into a plain
/// output that replaces: through [`Tile::drained_into`], which opens the scratch the bounce needs,
/// or where `copies` through [`Tile::copy_from`], which bounces the same way through the scratch it
/// is opened with.
#[cube(launch)]
fn fragment_matmul_into_a_short_window<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] copies: bool,
    #[define(E)] _output: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in &space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let mut c_cube = c.at(&cube);
        let opened =
            c_cube.accumulator::<E>(&a_cube, &b_cube, Instruction::Cmma, Semiring::SUM_PROD);
        // The drain opens the scratch the window needs; a copy is handed one.
        let mut acc = if comptime!(copies) {
            opened.with_scratch(Scratch::OneTile)
        } else {
            opened
        };
        acc.zero();
        let walk = cube.walk();
        let mut stages = Stages::smem(
            &walk,
            &a_cube,
            &b_cube,
            comptime!(StageStorage::Tiled {
                block: vec![(M, 16), (N, 16), (K, 16)],
                chunks: RowChunks::InOrder,
            }),
            1usize,
            None,
        );
        stages.pipelined(walk, |slot, stage| {
            slot.consume(|a_stage, b_stage| {
                let mut acc_stage = acc.at(stage);
                acc_stage.mma(a_stage, b_stage);
            });
        });
        if comptime!(copies) {
            c_cube.copy_from(&acc);
        } else {
            acc.drained_into(&c_cube);
        }
    }
}

/// Whether this device contracts `16×16×16` `f16` fragments summed in `f32`; reported rather than
/// silently passed where it cannot.
fn contracts_f16_fragments(client: &cubecl::client::Client) -> bool {
    let f16 = half::f16::elem_type_native();
    let f32 = f32::elem_type_native();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f16
            && cfg.b_type == f16
            && cfg.cd_type == f32
            && (cfg.m, cfg.n, cfg.k) == (16, 16, 16)
    });
    if !offered {
        ValidationResult::Skipped(
            "device has no 16x16x16 f16 cmma fragment summed in f32".to_string(),
        )
        .enforce();
    }
    offered
}

/// A `20 × 24` product in `16 × 16` boxes: every box but one overhangs the output, and the cells
/// past the edge are never written, so the output reads the product and nothing else.
#[test]
fn a_fragment_drains_into_a_window_the_edge_cuts_short() {
    short_window_matmul(false);
}

/// The same product stored with `copy_from`, which bounces a fragment into a short window just as
/// the drain does.
#[test]
fn a_fragment_copies_into_a_window_the_edge_cuts_short() {
    short_window_matmul(true);
}

fn short_window_matmul(copies: bool) {
    let client = cubecl::test_device().client();
    if !contracts_f16_fragments(&client) {
        return;
    }
    let (m, n, k) = (20usize, 24usize, 32usize);
    let input = half::f16::elem_type_native();
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(input)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(input)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 16), (N, 16), (K, 16)])
                .walk(&[(M, 1), (N, 1)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );
    // Bound through the launcher, which reads the overhang off the partitioning and masks the
    // output's writes past its edge.
    let bind = |binding, element, axes: &'static [Axis]| {
        launcher.arg(binding, element).axes(axes).build().unwrap()
    };
    fragment_matmul_into_a_short_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        bind(a_handle.clone().binding(), input, &[M, K]).arg(),
        bind(b_handle.clone().binding(), input, &[K, N]).arg(),
        bind(out.clone().binding(), out.dtype, &[M, N]).arg(),
        launcher.partitioning_arg(),
        copies,
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    assert_matches(&got, m, n, k);
}

// -- A fragment taller than its box -------------------------------------------------------------
//
// A decode's few rows under a manual-mma fragment of sixteen: each plane's fragment holds more rows
// than the cube's accumulator, which is sized to the box, and only the box's rows may land.

/// [`smem_split_matmul`] through manual-mma fragments: each plane contracts its slice of `K` in a
/// fragment the box's rows do not fill, and adds it into the cube's accumulator.
///
/// `canary` is shared memory declared after the accumulator, every cell the bits of `-0.0`: adding
/// anything into it as a float, even the fragment's zero rows, turns a cell `+0.0`. `touched`
/// counts, per cube, the cells that turned.
#[cube(launch)]
fn smem_split_fragment_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    touched: &mut Tensor<f32>,
    space: Partitioning,
    #[define(E)] _sum: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in &space {
        let mut c_cube = c.at(&cube);
        let sum = c_cube.smem_accumulation::<E>();
        let mut canary = Shared::<[u32]>::new_slice(CANARY);
        let mut cell = UNIT_POS as usize;
        while cell < CANARY {
            canary[cell] = NEGATIVE_ZERO;
            cell += CUBE_DIM as usize;
        }
        sync_cube();
        for plane in cube {
            let a_plane = a.at(&plane);
            let b_plane = b.at(&plane);
            let sink_plane = sum.sink.at(&plane);
            let mut partial = sink_plane.accumulator::<E>(
                &a_plane,
                &b_plane,
                comptime!(Instruction::Mma {
                    io: MmaIo::manual(),
                    operands: MmaOperands::Plain,
                }),
                Semiring::SUM_PROD,
            );
            partial.zero();
            partial.mma(&a_plane, &b_plane);
            partial.drained_into(&sink_plane);
        }
        sync_cube();
        c_cube.copy_from(&sum.source);
        if UNIT_POS == 0 {
            let mut turned = 0.0f32;
            for cell in 0..CANARY {
                if canary[cell] != NEGATIVE_ZERO {
                    turned += 1.0;
                }
            }
            touched[CUBE_POS] = turned;
        }
    }
}

/// Cells of the canary: past every row a sixteen-row fragment over one-row boxes reaches.
const CANARY: usize = 1024;

/// The bits of `-0.0`, held as an integer: a float literal of it is not kept by every backend.
const NEGATIVE_ZERO: u32 = 0x8000_0000;

/// Whether this device runs a `16×8×16` `f16` manual mma summed in `f32`; reported rather than
/// silently passed where it cannot.
fn runs_f16_manual_mma(client: &cubecl::client::Client) -> bool {
    let f16 = half::f16::elem_type_native();
    let f32 = f32::elem_type_native();
    let offered = client.properties().features.matmul.mma.iter().any(|cfg| {
        cfg.a_type == f16
            && cfg.b_type == f16
            && cfg.cd_type == f32
            && (cfg.m, cfg.n, cfg.k) == (16, 8, 16)
    });
    if !offered {
        ValidationResult::Skipped("device has no 16x8x16 f16 manual mma summed in f32".to_string())
            .enforce();
    }
    offered
}

/// Four rows under sixteen-row fragments, four planes splitting `K`: the fragments' twelve rows
/// past the box add into nothing. The accumulator holds the box's rows alone, so a row past them
/// landing would add past its end, out of the cube's shared memory.
#[test]
fn a_fragment_taller_than_its_box_adds_only_the_rows_of_its_box() {
    let client = cubecl::test_device().client();
    if !adds_atomically_in_shared_memory(&client) || !runs_f16_manual_mma(&client) {
        return;
    }
    let (m, n, k, planes) = (1usize, 8usize, 512usize, 4usize);
    let input = half::f16::elem_type_native();
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    // Stored `N`-major, as a lane run reads `B`: cell `(p, j)` at `j * k + p`.
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i % k * n + i / k) % 5) as f32 - 2.0)
        .collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(input)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![n, k])
        .dtype(input)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let cubes = n / 8;
    let touched = TestInput::builder(client.clone(), shape![cubes])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 16), (N, 8), (K, k / planes)])
                .walk_every(&[M])
                .planes(&[(K, planes)])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    );
    let bind = |binding, element, axes: &'static [Axis]| {
        launcher.arg(binding, element).axes(axes).build().unwrap()
    };
    smem_split_fragment_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        bind(a_handle.clone().binding(), input, &[M, K]).arg(),
        bind(b_handle.clone().binding(), input, &[N, K]).arg(),
        bind(out.clone().binding(), out.dtype, &[M, N]).arg(),
        touched.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    assert_matches(&got, m, n, k);
    let touched = HostData::from_tensor_handle(&client, touched, HostDataType::F32);
    for cube in 0..cubes {
        assert_eq!(
            touched.get_f32(&[cube]),
            0.0,
            "cube {cube}: rows past the box landed past the accumulator"
        );
    }
}
