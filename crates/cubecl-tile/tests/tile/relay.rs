//! Destinations that add. A relay's cubes, the runs of a contraction split across them, take their
//! turns at one carry and hand the sum on to the output through a cast; an atomic destination
//! takes launches adding into one buffer, a pitched one included; and a window of memory lands in a
//! narrower output through a cast, as a relay's last turn copies it.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, StridedLayout, TestInput, ValidationResult};
use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, Type},
    prelude::*,
    zspace::shape,
};
use cubecl_tile::kind::Write;
use cubecl_tile::launch::{Buffered, Destination, DestinationLaunch};
use cubecl_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The leaf's register block in the relay's contraction.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// Rows and columns every case holds, cut by tiles of [`TILE`] that overhang both.
const ROWS: usize = 5;
const COLS: usize = 29;
const TILE: (usize, usize) = (2, 8);

/// The row stride of a pitched output: rows 29 wide padded to 32, so the buffer holds lines past
/// the shape's product.
const PITCH: usize = 32;

/// `out` written with `input`, as the destination's write says: replaced or added atomically.
#[cube(launch)]
fn store<E: Float, V: Size, O: Destination>(
    input: &TileArg<'_>,
    out: &O::Arg<E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(&space);
    let mut dst = O::tile::<E, V>(out, &space);
    dst.copy_from(&src);
}

/// `out` written with `input` cast to the output's element.
#[cube(launch)]
fn store_cast(input: &TileArg<'_>, out: &TileArg<'_>, space: Partitioning) {
    let src = input.tile(&space);
    let mut dst = out.tile(&space);
    dst.copy_cast_from(&src);
}

fn launcher() -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, TILE.0), (COL, TILE.1)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        Form::Static,
    )
}

/// What the output held before the first launch: not zero, so a write that replaced the cell
/// rather than adding into it shows.
fn start(row: usize, col: usize) -> f32 {
    ((row * 7 + col) % 11) as f32 - 5.0
}

/// `launches` launches, each adding `row · COLS + col` into an output that started at [`start`],
/// laid out `layout`, as `write` says; the output read back.
fn add_launches(write: Write, layout: StridedLayout, launches: usize) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = launcher();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let initial = (0..ROWS * COLS)
        .map(|i| start(i / COLS, i % COLS))
        .collect();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .layout(layout)
        .custom(initial)
        .generate_without_host_data();
    for _ in 0..launches {
        let src = launcher
            .arg(input.clone().binding(), dtype)
            .axes(&[ROW, COL])
            .build()
            .unwrap();
        let bound = launcher
            .arg(output.clone().binding(), dtype)
            .axes(&[ROW, COL])
            .build()
            .unwrap();
        store::launch::<Buffered>(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            1,
            src.arg(),
            Buffered::arg((bound, write)),
            launcher.partitioning_arg(),
            dtype,
        );
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

fn assert_added(got: &HostData, launches: usize, what: &str) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let expected = start(row, col) + (launches * (row * COLS + col)) as f32;
            assert_eq!(
                got.get_f32(&[row, col]),
                expected,
                "{what}: [{row}, {col}] is not what it started at plus {launches} launches' values"
            );
        }
    }
}

/// Three launches adding atomically into one buffer: the buffer's own values plus each launch's,
/// the tiles past the problem's edge written only where they lie inside.
#[test]
fn launches_adding_atomically_into_one_buffer_sum() {
    let got = add_launches(Write::Accumulate, StridedLayout::RowMajor, 3);
    assert_added(&got, 3, "added atomically");
}

/// A pitched buffer holds lines past its shape's product; an atomic destination writes the rows
/// there like any other, rather than dropping every write past `rows · cols`.
#[test]
fn a_pitched_buffer_takes_every_row_added_into_it() {
    let pitched = StridedLayout::Explicit(vec![PITCH, 1]);
    assert_added(
        &add_launches(Write::Accumulate, pitched, 2),
        2,
        "added atomically, pitched",
    );
}

/// A window of `f32` memory copied into an `f16` output of the same box by the whole cube, the
/// tiles past the problem's edge copied only where they lie inside.
#[test]
fn a_memory_window_lands_in_a_narrower_output_through_a_cast() {
    let client = cubecl::test_device().client();
    let (wide, narrow) = (f32::elem_type_native(), half::f16::elem_type_native());
    let launcher = launcher();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(wide)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(narrow)
        .zeros()
        .generate_without_host_data();
    let src = launcher
        .arg(input.binding(), wide)
        .axes(&[ROW, COL])
        .build()
        .unwrap();
    let dst = launcher
        .arg(output.clone().binding(), narrow)
        .axes(&[ROW, COL])
        .build()
        .unwrap();
    store_cast::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        src.arg(),
        dst.arg(),
        launcher.partitioning_arg(),
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            // Every value here is an integer under 2048, which f16 holds exactly.
            assert_eq!(
                got.get_f32(&[row, col]),
                (row * COLS + col) as f32,
                "the cast copy is wrong at [{row}, {col}]"
            );
        }
    }
}

// -- A relay --------------------------------------------------------------------------------------

/// `c = a · b` in register blocks, `K` split across the cubes of each box: each contracts its run
/// into an accumulator opened on the relayed box, takes its turn at the `f32` carry, and the last
/// casts the sum into the `f16` output.
#[cube(launch)]
fn relayed_split_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    carry: &TileArg<'_>,
    out: &TileArg<'_>,
    turns: &[Atomic<u32>],
    space: Partitioning,
) {
    let relay = carry.relay::<Const<1>>(&space, turns);
    // Uniform across the cube, so it leaves before any barrier.
    if !relay.has_turn() {
        terminate!();
    }
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut out = out.tile(&space);
    let c = relay.tile();
    let acc = c.accumulator::<f32>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: REGISTER_BLOCK
        }),
        Semiring::SUM_PROD,
    );
    for cube in &space {
        for step in cube.walk() {
            let mut acc_step = acc.at(&step);
            acc_step.mma(&a.at(&step), &b.at(&step));
        }
    }
    relay.take();
    acc.drained_into(&c);
    relay.pass(&mut out);
}

/// Whether this device hands one cube's writes to another within a dispatch and stores and adds
/// `u32`s atomically, the two things the relay needs; reported rather than silently passed.
pub(super) fn relays(client: &cubecl::client::Client) -> bool {
    let hands_off = client.properties().features.device_memory_scope;
    let usage = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)));
    let counts = usage.contains(AtomicUsage::LoadStore) && usage.contains(AtomicUsage::Add);
    if !(hands_off && counts) {
        ValidationResult::Skipped(
            "device has no device-scope storage sync or no u32 atomic store and add".to_string(),
        )
        .enforce();
    }
    hands_off && counts
}

/// `a · b` over an `m × n` output of `8 × 8` boxes, `K` split across `splits` cubes a box, run
/// `launches` times over the same carry, turn counters and output; the output and the counters
/// read back.
fn run_relayed_split(
    (m, n, k): (usize, usize, usize),
    splits: usize,
    launches: usize,
) -> (HostData, Vec<u32>) {
    let client = cubecl::test_device().client();
    let (f32_ty, f16_ty) = (f32::elem_type_native(), half::f16::elem_type_native());
    let edge = 8usize;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let a = TestInput::builder(client.clone(), shape![m, k])
        .dtype(f32_ty)
        .custom(a)
        .generate_without_host_data();
    let b = TestInput::builder(client.clone(), shape![k, n])
        .dtype(f32_ty)
        .custom(b)
        .generate_without_host_data();
    // The carry arrives holding anything: the first turn replaces it.
    let carry = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32_ty)
        .custom(vec![1.0e6; m * n])
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f16_ty)
        .zeros()
        .generate_without_host_data();

    // One box a cube along `M` and `N`, `K` in runs of whole steps.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[K])
                .cubes(&[M, N, K])
                .across(K, splits)
                .build(),
        ),
        Form::Static,
    );
    let counters = launcher.partitioning().relay_counters();
    let turns = client.create_from_slice(u32::as_bytes(&vec![0u32; counters]));

    for _ in 0..launches {
        relayed_split_matmul::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(
                a.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[M, K]),
            ),
            TileArgLaunch::new(
                b.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[K, N]),
            ),
            TileArgLaunch::new(
                carry.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[M, N]),
            ),
            TileArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                f16_ty,
                1,
                TileSpec::direct(&[M, N]),
            ),
            unsafe { BufferArg::from_raw_parts(turns.clone(), counters) },
            launcher.partitioning_arg(),
        );
    }
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let turns = u32::from_bytes(&client.read_one_unchecked(turns)).to_vec();
    (got, turns)
}

fn assert_relayed(got: &HostData, (m, n, k): (usize, usize, usize), what: &str) {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    for i in 0..m {
        for j in 0..n {
            // Small integers, each within f16's exact range.
            let want: f32 = (0..k).map(|p| a(i, p) * b(p, j)).sum();
            assert_eq!(got.get_f32(&[i, j]), want, "{what}: ({i}, {j})");
        }
    }
}

/// Every split of `K` sums to the whole contraction, box by box, however many cubes take a turn;
/// the counters are left at zero.
#[test]
fn a_relay_sums_every_run_of_a_split_contraction() {
    let client = cubecl::test_device().client();
    if !relays(&client) {
        return;
    }
    let dims = (16usize, 16usize, 64usize);
    for splits in [1usize, 2, 4, 8] {
        let (got, turns) = run_relayed_split(dims, splits, 1);
        assert_relayed(&got, dims, &format!("{splits} splits"));
        assert!(turns.iter().all(|&t| t == 0), "{splits} splits: {turns:?}");
    }
}

/// Five stages over four cubes run two at a time, so the fourth cube's run is empty: it takes no
/// turn, and the third's is the last.
#[test]
fn a_cube_whose_run_is_empty_takes_no_turn() {
    let client = cubecl::test_device().client();
    if !relays(&client) {
        return;
    }
    let dims = (8usize, 16usize, 40usize);
    let (got, turns) = run_relayed_split(dims, 4, 1);
    assert_relayed(&got, dims, "an empty run");
    assert!(turns.iter().all(|&t| t == 0), "{turns:?}");
}

/// The counters a relay leaves at zero take the next launch as they took the first.
#[test]
fn a_relay_runs_again_on_the_counters_it_left() {
    let client = cubecl::test_device().client();
    if !relays(&client) {
        return;
    }
    let dims = (16usize, 8usize, 32usize);
    let (got, turns) = run_relayed_split(dims, 4, 3);
    assert_relayed(&got, dims, "a third launch");
    assert!(turns.iter().all(|&t| t == 0), "{turns:?}");
}
