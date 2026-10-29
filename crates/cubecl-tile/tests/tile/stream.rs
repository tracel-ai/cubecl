//! Distributing a level's work as one index: the hypercube scope taking every step of the
//! contraction as dependent work.
//!
//! Distributing each axis on its own gives a cube the product of its per-axis runs, which is a box
//! of the grid. A share of the work is not a box: it is a range of the index the axes make
//! together, and may start in one tile and end in another; no box of a four by two grid holds three
//! regions.
//!
//! [`Walk::portion`] is that range, and the [`Portion`] it returns the walk over it. The axes of the
//! distributed work stay `Sequential`, so the walk's counts are the whole grid and its flat index
//! carries every coordinate; an instance's run is `base` and `steps` into it, both runtime values.
//!
//! The first tests here are the assignment on a copy, with no contraction and nothing partial:
//! they prove the runs cover the grid exactly once, and that a run starting late reads the
//! regions it was given.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    client::Client,
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubecl_tile::kind::Boundary;
use cubecl_tile::launch::AccumulateArg;
use cubecl_tile::launch::AccumulateArgLaunch;
use cubecl_tile::launch::BoundaryPolicy;
use cubecl_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 12;
const COLS: usize = 10;
/// Tile edges: a 4 by 2 grid, so 8 regions in the flat step space.
const TILE_ROWS: usize = 3;
const TILE_COLS: usize = 5;
const REGIONS: usize = (ROWS / TILE_ROWS) * (COLS / TILE_COLS);

/// Each cube copies the contiguous run of regions `[pos · total / cubes, (pos + 1) · total /
/// cubes)`. Written as two divisions rather than a per-cube share so the ragged case needs no
/// branch: the runs abut, cover the grid once, and differ in length by at most one.
#[cube(launch)]
fn copy_run(
    src: &TileArg<'_>,
    dst: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] cubes: usize,
) {
    let src = src.tile(&space);
    let dst = dst.tile(&space);
    let walk = dst.over(&level);
    let total = walk.total();
    let pos = CUBE_POS_X as usize;

    let start = pos * total / cubes;
    let end = (pos + 1) * total / cubes;

    for region in walk.range(start, end - start) {
        dst.at(&region).copy_from(&src.at(&region));
    }
}

/// One cube copying the stated run and nothing else, so a `base` that is dropped copies the wrong
/// regions rather than the same ones in another order.
#[cube(launch)]
fn copy_one_run(
    src: &TileArg<'_>,
    dst: &TileArg<'_>,
    #[comptime] start: usize,
    #[comptime] steps: usize,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let src = src.tile(&space);
    let dst = dst.tile(&space);
    let walk = dst.over(&level);

    // Stated at launch but taken as runtime values: a window whose bounds evaluate to constants
    // would prove the decode only for the case the compiler could have unrolled.
    for region in walk.range(comptime!(start).runtime(), comptime!(steps).runtime()) {
        dst.at(&region).copy_from(&src.at(&region));
    }
}

struct Harness {
    client: Client,
    dtype: ElemType,
    launcher: Launcher,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: cubecl::test_device().client(),
            dtype: f32::elem_type_native(),
            launcher: implied(
                &cubecl::test_device().client(),
                Partitioning::new(
                    Space::new(&[(ROW, ROWS), (COL, COLS)]),
                    Levels::leaf(&[(ROW, TILE_ROWS), (COL, TILE_COLS)])
                        .walk_every(&[ROW, COL])
                        .build(),
                ),
                Form::Static,
            ),
        }
    }

    fn source(&self) -> (TensorHandle, HostData) {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .arange()
            .generate_with_f32_host_data()
    }

    fn destination(&self) -> TensorHandle {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle) -> HostData {
        HostData::from_tensor_handle(&self.client, output, HostDataType::F32)
    }

    /// `src` and `dst` as the launch arguments, served at the harness's element.
    fn tile_args(&self, src: &TensorHandle, dst: &TensorHandle) -> (TileArgLaunch, TileArgLaunch) {
        (
            TileArgLaunch::new(
                src.clone().binding().into_tensor_arg(),
                self.dtype,
                1,
                TileSpec::direct(&[ROW, COL]),
            ),
            TileArgLaunch::new(
                dst.clone().binding().into_tensor_arg(),
                self.dtype,
                1,
                TileSpec::direct(&[ROW, COL]),
            ),
        )
    }
}

/// `cubes` runs between them copy every cell exactly once. A run that overlapped its neighbour
/// would still pass on a copy, so the case that matters is the cell no run claimed: it stays at
/// the destination's zero.
fn runs_cover_the_grid(cubes: usize) {
    let h = Harness::new();
    let (src, want) = h.source();
    let dst = h.destination();
    let (src_arg, dst_arg) = h.tile_args(&src, &dst);

    copy_run::launch(
        &h.client,
        CubeCount::Static(cubes as u32, 1, 1),
        h.launcher.cube_dim(),
        src_arg,
        dst_arg,
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        cubes,
    );

    let got = h.read(dst);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                got.get_f32(&[row, col]),
                want.get_f32(&[row, col]),
                "{cubes} cubes: cell ({row}, {col}) was not copied exactly once"
            );
        }
    }
}

#[test]
fn runs_dividing_the_grid_cover_it_once() {
    // 8 regions over 4 cubes: two each.
    runs_cover_the_grid(4);
}

#[test]
fn runs_that_do_not_divide_the_grid_still_cover_it_once() {
    // 8 regions over 3 cubes: 2, 3, 3. The case a per-axis distribute cannot express, since no
    // rectangle of a 4 by 2 grid has three regions in it.
    runs_cover_the_grid(3);
    // And one region each for five of eight, which leaves three cubes with an empty run.
    runs_cover_the_grid(REGIONS + 2);
}

#[test]
fn a_run_starting_late_copies_the_regions_it_was_given() {
    let h = Harness::new();
    let (src, want) = h.source();
    let dst = h.destination();
    let (src_arg, dst_arg) = h.tile_args(&src, &dst);

    // Regions 3 and 4 of the row-major walk: the second half of row band 1, and the first half of
    // row band 2. A rectangle cannot name that pair either.
    let (start, steps) = (3usize, 2usize);
    copy_one_run::launch(
        &h.client,
        CubeCount::Static(1, 1, 1),
        h.launcher.cube_dim(),
        src_arg,
        dst_arg,
        start,
        steps,
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
    );

    let got = h.read(dst);
    let cols_per_band = COLS / TILE_COLS;
    for row in 0..ROWS {
        for col in 0..COLS {
            let region = (row / TILE_ROWS) * cols_per_band + col / TILE_COLS;
            let copied = region >= start && region < start + steps;
            let expect = match copied {
                true => want.get_f32(&[row, col]),
                false => 0.0,
            };
            assert_eq!(
                got.get_f32(&[row, col]),
                expect,
                "cell ({row}, {col}) in region {region}"
            );
        }
    }
}

// -- The contraction ---------------------------------------------------------
//
// The assignment above, over a matmul. `K` is not cut at cube scope and no axis is: the line runs
// over the output's tiles and each tile's `K` blocks together, and a cube takes a run of it. A run
// spans whole tiles inside, partial ones at each end; cubes then share cells, which the sink sums.

const MM: Axis = Axis(2);
const NN: Axis = Axis(3);
const KK: Axis = Axis(4);
const BB: Axis = Axis(5);

/// The leaf's register block, held fixed across every kernel here.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// One step of the contraction into `acc`'s cell at `region`.
#[cube]
fn contract(acc: &Tile, a: &Tile, b: &Tile, region: &Region) {
    let mut acc_cell = acc.at(region);
    acc_cell.mma(&a.at(region), &b.at(region));
}

/// The streamed contraction: each cube takes its run of the joint index over the output's tiles
/// and their contraction, opens a register accumulator per output region it touches, sums that
/// region's part of the run, and drains once through the atomic sink.
///
/// `inner` is the level the run is counted in; `leaf` a level below it, where the run's step is
/// itself walked (a unit's run of `K`), or none where the step is the contraction's own.
#[cube(launch)]
fn stream_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] leaf: Option<Level>,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    let portion = space.over(&outer).portion(comptime!(inner.clone()));
    for i in 0..portion.touched() {
        let region = portion.region(i);
        let (from, steps) = portion.steps(i);
        let c_region = c.at(&region);
        let a_region = a.at(&region);
        let b_region = b.at(&region);
        let acc = c_region.accumulator::<E>(
            &a_region,
            &b_region,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for cell in region.over(&inner).range(from, steps) {
            match comptime!(leaf.clone()) {
                Some(leaf) => {
                    for step in cell.over(&leaf) {
                        contract(&acc, &a_region, &b_region, &step);
                    }
                }
                None => contract(&acc, &a_region, &b_region, &cell),
            }
        }
        acc.drained_into(&c_region);
    }
}

/// [`stream_matmul`] with the right operand staged into shared memory under the share: each
/// region runs the level below through its own stages, so what a share does inside a region is
/// what any walk does.
#[cube(launch)]
fn stream_matmul_staged_rhs<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    let portion = space.over(&outer).portion(comptime!(inner.clone()));
    for i in 0..portion.touched() {
        let region = portion.region(i);
        let (from, steps) = portion.steps(i);
        let c_region = c.at(&region);
        let a_region = a.at(&region);
        let b_region = b.at(&region);
        let acc = c_region.accumulator::<E>(
            &a_region,
            &b_region,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        let cells = region.over(&inner).range(from, steps);
        let mut stages =
            Stages::smem_single(&cells, &b_region, StageStorage::Strided, 1usize, None);
        stages.pipelined(cells, |slot, cell| {
            let mut acc_cell = acc.at(cell);
            let a_cell = a_region.at(cell);
            slot.consume(|b_s| {
                acc_cell.mma(&a_cell, b_s);
            });
        });
        acc.drained_into(&c_region);
    }
}

/// [`stream_matmul`] over the first `prefix` steps of every tile's contraction alone, the runs
/// sharing that prefix ([`Walk::portion_of`]): the steps past it are no run's.
#[cube(launch)]
fn stream_matmul_prefix<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    prefix: usize,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    let portion = space.over(&outer).portion_of(prefix);
    for i in 0..portion.touched() {
        let region = portion.region(i);
        let (from, steps) = portion.steps(i);
        let c_region = c.at(&region);
        let a_region = a.at(&region);
        let b_region = b.at(&region);
        let acc = c_region.accumulator::<E>(
            &a_region,
            &b_region,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for cell in region.over(&inner).range(from, steps) {
            contract(&acc, &a_region, &b_region, &cell);
        }
        acc.drained_into(&c_region);
    }
}

/// The tile the contraction accumulates in, and the block it walks `K` in.
const TILE_M: usize = 4;
const TILE_N: usize = 4;
const BLOCK_K: usize = 4;

/// Where the right operand is read from at the level *below* the distribution, which is the one
/// that walks a share step by step and the only one that can stage anything.
#[derive(Clone, Copy)]
enum RhsStage {
    InPlace,
    Smem,
}

/// `a · b` with the work shared between `runs` cubes, added atomically into a zeroed output.
fn run_shared_index(m: usize, n: usize, k: usize, runs: usize, rhs: RhsStage) -> HostData {
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
    // Zeroed by the launch: no cube owns a cell, so none of them may seed one.
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(MM, m), (NN, n), (KK, k)]),
            Levels::leaf(&[(MM, TILE_M), (NN, TILE_N), (KK, BLOCK_K)])
                .walk(&[(MM, 1), (NN, 1), (KK, k / BLOCK_K)])
                .cubes(&[MM, NN, KK])
                .shared_by(runs)
                .build(),
        ),
        Form::Static,
    );

    match rhs {
        RhsStage::InPlace => stream_matmul::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                1,
                TileSpec::direct(&[MM, NN]),
            ),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            None,
            dtype,
        ),
        RhsStage::Smem => stream_matmul_staged_rhs::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                1,
                TileSpec::direct(&[MM, NN]),
            ),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            dtype,
        ),
    }

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// `a · b` over the first `prefix` blocks of `K` alone, shared between `runs` cubes, added
/// atomically into a zeroed output.
fn run_shared_prefix(m: usize, n: usize, k: usize, runs: usize, prefix: usize) -> HostData {
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

    // The cubes take whole output tiles and share the line of them, each tile `k / BLOCK_K` steps.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(MM, m), (NN, n), (KK, k)]),
            Levels::leaf(&[(MM, TILE_M), (NN, TILE_N), (KK, BLOCK_K)])
                .walk(&[(MM, 1), (NN, 1), (KK, k / BLOCK_K)])
                .cubes(&[MM, NN])
                .shared_by(runs)
                .build(),
        ),
        Form::Static,
    );

    stream_matmul_prefix::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[MM, KK]),
        ),
        TileArgLaunch::new(
            b_handle.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[KK, NN]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[MM, NN]),
        ),
        launcher.partitioning_arg(),
        prefix,
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// The same contraction on the host, from the same seeds.
fn reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    reference_over(m, n, k, k)
}

/// [`reference`] summed over the first `contracted` of `K` alone.
fn reference_over(m: usize, n: usize, k: usize, contracted: usize) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..contracted).map(|p| a(i, p) * b(p, j)).sum();
        }
    }
    out
}

/// Every run count computes the same contraction. The interesting ones are those that do not
/// divide the line: their runs start and end inside a tile's contraction, which is what makes
/// this one shared index rather than a split of `K`.
fn shared_index_agrees_with_the_whole(cubes: usize) {
    let (m, n, k) = (8usize, 8usize, 16usize);
    let got = run_shared_index(m, n, k, cubes, RhsStage::InPlace);
    let want = reference(m, n, k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "{cubes} runs: at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The control: one run is the whole line, so nothing is partial and nothing is summed across
/// cubes. A stream that only works when it is split is not working.
#[test]
fn a_stream_of_one_run_is_the_whole_contraction() {
    if !adds_atomically() {
        return;
    }
    shared_index_agrees_with_the_whole(1);
}

/// Runs that end on a tile boundary: the same work a split of `K` would do, reached by distributing
/// a line rather than by cutting an axis.
#[test]
fn runs_that_end_on_a_tile_do_the_split_a_cut_would() {
    if !adds_atomically() {
        return;
    }
    // 4 tiles of 4 K blocks: 16 steps of line over 4 and 8 runs.
    shared_index_agrees_with_the_whole(4);
    shared_index_agrees_with_the_whole(8);
}

/// Runs that do not: 16 steps over 3, 5 and 7 leaves every run straddling a tile boundary, so a
/// cube seeds a fresh accumulator part way through a tile's contraction and sums a slice that no
/// cut of `K` could hand it.
#[test]
fn runs_that_straddle_a_tile_boundary_still_sum_to_the_whole() {
    if !adds_atomically() {
        return;
    }
    shared_index_agrees_with_the_whole(3);
    shared_index_agrees_with_the_whole(5);
    shared_index_agrees_with_the_whole(7);
    // More runs than the line has steps, so some cubes get nothing at all.
    shared_index_agrees_with_the_whole(24);
}

/// Runs sharing a prefix of every tile's steps, which the launch states: the steps past it are no
/// run's, so the runs split the prefix evenly and every tile sums its prefix alone, whether a run
/// starts and ends inside a tile or holds several, and where some runs hold nothing.
#[test]
fn runs_share_the_prefix_of_every_tile_the_launch_states() {
    if !adds_atomically() {
        return;
    }
    // 4 tiles of 6 K blocks, of which the first 4, then 1, are shared.
    let (m, n, k) = (8usize, 8usize, 24usize);
    for prefix in [4usize, 1] {
        let want = reference_over(m, n, k, prefix * BLOCK_K);
        for runs in [1usize, 3, 5, 24] {
            let got = run_shared_prefix(m, n, k, runs, prefix);
            for i in 0..m {
                for j in 0..n {
                    let have = got.get_f32(&[i, j]);
                    let want = want[i * n + j];
                    assert!(
                        (have - want).abs() < 1e-3,
                        "{runs} runs over a prefix of {prefix}: at ({i}, {j}): got {have}, want \
                         {want}"
                    );
                }
            }
        }
    }
}

/// The drain adds atomically, so a device without a float atomic add cannot run any of this.
fn adds_atomically() -> bool {
    let client = cubecl::test_device().client();
    let adds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add);
    if !adds {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
    }
    adds
}

/// An operand staged under the distribution. A share is walked region by region, and each region
/// runs the level below through its own stages, so what a share does inside a region is what any
/// walk does: nothing about staging changes because the regions arrived as a share.
#[test]
fn an_operand_stages_under_a_share_as_it_does_under_a_walk() {
    if !adds_atomically() {
        return;
    }
    let (m, n, k) = (8usize, 8usize, 16usize);
    let want = reference(m, n, k);
    // Shares that straddle a tile boundary, so the staged operand is read from part way through
    // a tile's contraction as well as from its start.
    for runs in [1usize, 3, 5] {
        let got = run_shared_index(m, n, k, runs, RhsStage::Smem);
        for i in 0..m {
            for j in 0..n {
                let have = got.get_f32(&[i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-3,
                    "{runs} shares, rhs staged: at ({i}, {j}): got {have}, want {want}"
                );
            }
        }
    }
}

/// Two scopes sharing one contraction: the cubes take shares of the work, and inside a cube the
/// plane's units cut `K` between them and meet in registers. The share is counted in the steps
/// the units take *together*, so a cube's slice is the same size however many units cover a step.
#[test]
fn cubes_take_shares_while_the_units_cut_k_between_them() {
    let client = cubecl::test_device().client();
    if !adds_atomically() {
        return;
    }
    let dtype = f32::elem_type_native();
    let plane_size = client.properties().hardware.plane_size_max as usize;
    // Two steps of `K` per unit, walked under the units, so a cube's share is counted in
    // something longer than one step of the contraction.
    let (m, n, k) = (8usize, 8usize, 2 * plane_size);
    let want = reference(m, n, k);

    // 4 output tiles of one unit tile each: 4 shares of work, over fewer cubes and more.
    for runs in [1usize, 3, 5] {
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
                Space::new(&[(MM, m), (NN, n), (KK, k)]),
                Levels::leaf(&[(MM, TILE_M), (NN, TILE_N), (KK, 1)])
                    .walk(&[(KK, k / plane_size)])
                    .units(&[(KK, plane_size)])
                    .cubes(&[MM, NN, KK])
                    .shared_by(runs)
                    .build(),
            ),
            Form::Static,
        );

        stream_matmul::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                1,
                TileSpec::direct(&[MM, NN]),
            ),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            Some(launcher.partitioning().level(2)),
            dtype,
        );

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for i in 0..m {
            for j in 0..n {
                let have = got.get_f32(&[i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-2,
                    "{runs} shares over cubes, K over {plane_size} plane_size: at ({i}, {j}): got {have}, \
                     want {want}"
                );
            }
        }
    }
}

// -- Merged by the last arrival ---------------------------------------------------------------------
//
// The same line, with no float atomic and no cube waiting on another: a cube holding a tile whole drains
// it into the output, and of the cubes holding parts of one, each parks its part in a slot and the
// last to count itself in adds them in the order of their runs and casts the total into the output.

/// The streamed contraction merged by [`LastArrival`](cubecl_tile::launch::LastArrival): each cube
/// contracts its part of every tile its run touches into a register accumulator opened on its slot,
/// then hands it on, cast to the output's element `O`.
#[cube(launch)]
fn merged_stream_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    slots: &TileArg<'_>,
    out: &TileArg<'_>,
    arrivals: &[Atomic<u32>],
    space: Partitioning,
    #[comptime] steps: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut out = out.tile(&space);
    let portion = space.walk().portion(comptime!(steps.clone()));
    for i in 0..portion.touched() {
        let region = portion.region(i);
        let (from, count) = portion.steps(i);
        let merge = slots.last_arrival::<Const<1>>(&space, &portion, i, arrivals);
        let (a_region, b_region) = (a.at(&region), b.at(&region));
        let acc = merge.tile().accumulator::<f32>(
            &a_region,
            &b_region,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for cell in region.over(&steps).range(from, count) {
            contract(&acc, &a_region, &b_region, &cell);
        }
        merge.hand_on(&acc, &mut out);
    }
}

/// `out[batch] = a[batch] · b` for every batch, `b` shared by all of them.
#[derive(Clone, Copy, Debug)]
struct Product {
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl Product {
    /// One batch of `m × n × k`.
    fn single(m: usize, n: usize, k: usize) -> Self {
        Product {
            batches: 1,
            m,
            n,
            k,
        }
    }

    /// The inputs of the integer cases: small integers, so every sum is exact in `f16`.
    fn integers(&self) -> (Vec<f32>, Vec<f32>) {
        let a = (0..self.batches * self.m * self.k)
            .map(|i| (i % 7) as f32 - 3.0)
            .collect();
        let b = (0..self.k * self.n).map(|i| (i % 5) as f32 - 2.0).collect();
        (a, b)
    }
}

/// `a · b` with every tile's blocks shared between `cubes` cubes and merged by the last arrival,
/// launched `launches` times over the same slots, counters and output, an output of `O`: the
/// output after each launch, and the counters read back.
fn run_merged_stream<O: Float + CubeElement>(
    product: Product,
    (a, b): (Vec<f32>, Vec<f32>),
    cubes: usize,
    launches: usize,
) -> (Vec<HostData>, Vec<u32>) {
    let Product { batches, m, n, k } = product;
    let client = cubecl::test_device().client();
    let (f32_ty, out_ty) = (f32::elem_type_native(), O::elem_type_native());
    let a = TestInput::builder(client.clone(), shape![batches, m, k])
        .dtype(f32_ty)
        .custom(a)
        .generate_without_host_data();
    let b = TestInput::builder(client.clone(), shape![k, n])
        .dtype(f32_ty)
        .custom(b)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![batches, m, n])
        .dtype(out_ty)
        .zeros()
        .generate_without_host_data();

    // The cube level hands out one batch a box, so a run's steps cross from one batch's tiles into
    // the next's.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(BB, batches), (MM, m), (NN, n), (KK, k)]),
            Levels::leaf(&[(MM, TILE_M), (NN, TILE_N), (KK, BLOCK_K)])
                .walk(&[(MM, 1), (NN, 1), (KK, k / BLOCK_K)])
                .cubes(&[MM, NN, KK])
                .shared_by(cubes)
                .batches(&[BB])
                .build(),
        ),
        Form::Static,
    );
    // The slots arrive holding anything: a part replaces what its slot held.
    let slots = launcher.partitioning().arrival_slots(&[BB, MM, NN]);
    let held = vec![1.0e6; slots.num_elements()];
    let slots = TestInput::builder(client.clone(), slots)
        .dtype(f32_ty)
        .custom(held)
        .generate_without_host_data();
    let counters = launcher.partitioning().arrival_counters();
    let arrivals = client.create_from_slice(u32::as_bytes(&vec![0u32; counters]));
    // Checked on every axis: a tile past the output's edge reads zero there and writes nothing.
    let spec =
        |axes: &[Axis]| TileSpec::direct(axes).boundary(BoundaryPolicy::Every(Boundary::Zero));
    let arg = |handle: &TensorHandle, axes: &[Axis]| {
        TileArgLaunch::new(
            handle.clone().binding().into_tensor_arg(),
            handle.dtype,
            1,
            spec(axes),
        )
    };
    let mut outputs = Vec::with_capacity(launches);
    for _ in 0..launches {
        merged_stream_matmul::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            arg(&a, &[BB, MM, KK]),
            arg(&b, &[KK, NN]),
            arg(&slots, &[BB, MM, NN]),
            arg(&out, &[BB, MM, NN]),
            unsafe { BufferArg::from_raw_parts(arrivals.clone(), counters) },
            launcher.partitioning_arg(),
            launcher.partitioning().level(1),
        );
        outputs.push(HostData::from_tensor_handle(
            &client,
            out.clone(),
            HostDataType::F32,
        ));
    }
    let arrivals = u32::from_bytes(&client.read_one_unchecked(arrivals)).to_vec();
    (outputs, arrivals)
}

/// Whether this device hands one cube's writes to another within a dispatch and adds `u32`s
/// atomically, the two things a merge by the last arrival needs; reported rather than silently
/// passed.
fn merges_on_arrival(client: &Client) -> bool {
    let hands_off = client.properties().features.device_memory_scope;
    let counts = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)))
        .contains(AtomicUsage::Add);
    if !(hands_off && counts) {
        ValidationResult::Skipped(
            "device has no device-scope storage sync or no u32 atomic add".to_string(),
        )
        .enforce();
    }
    hands_off && counts
}

fn assert_merged(got: &HostData, product: Product, what: &str) {
    let Product { batches, m, n, k } = product;
    let (a, b) = product.integers();
    for batch in 0..batches {
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k)
                    .map(|p| a[(batch * m + i) * k + p] * b[p * n + j])
                    .sum();
                // Small integers, each within f16's exact range.
                assert_eq!(
                    got.get_f32(&[batch, i, j]),
                    want,
                    "{what}: ({batch}, {i}, {j})"
                );
            }
        }
    }
}

/// Every count of cubes sums to the whole contraction: runs ending on a tile's edge, runs
/// straddling one, a run a tile, and more cubes than tiles, so that many share each tile and the
/// last cubes hold nothing. Each tile's counter is left at zero.
#[test]
fn the_last_arrival_merges_every_tile_of_a_shared_contraction() {
    let client = cubecl::test_device().client();
    if !merges_on_arrival(&client) {
        return;
    }
    // Twelve tiles of five blocks: sixty blocks in all.
    let product = Product::single(12, 16, 20);
    for cubes in [1usize, 2, 5, 7, 11, 12, 13, 25, 59, 60, 61, 97] {
        let (got, arrivals) = run_merged_stream::<half::f16>(product, product.integers(), cubes, 1);
        assert_merged(&got[0], product, &format!("{cubes} cubes"));
        assert!(
            arrivals.iter().all(|&a| a == 0),
            "{cubes} cubes: {arrivals:?}"
        );
    }
}

/// Tiles past the output's edge: a part parks its whole box, and only what lies inside the output
/// lands there.
#[test]
fn the_last_arrival_merges_tiles_that_overhang_the_output() {
    let client = cubecl::test_device().client();
    if !merges_on_arrival(&client) {
        return;
    }
    let product = Product::single(10, 14, 20);
    for cubes in [3usize, 8, 23] {
        let (got, arrivals) = run_merged_stream::<half::f16>(product, product.integers(), cubes, 1);
        assert_merged(&got[0], product, &format!("{cubes} cubes"));
        assert!(
            arrivals.iter().all(|&a| a == 0),
            "{cubes} cubes: {arrivals:?}"
        );
    }
}

/// Three batches of a product, each its own tiles: runs cross from one batch's tiles into the
/// next's, and a slot is one box whichever batch its tile is in.
#[test]
fn the_last_arrival_merges_the_tiles_of_every_batch() {
    let client = cubecl::test_device().client();
    if !merges_on_arrival(&client) {
        return;
    }
    let product = Product {
        batches: 3,
        m: 8,
        n: 12,
        k: 20,
    };
    for cubes in [4usize, 7, 18, 31] {
        let (got, arrivals) = run_merged_stream::<half::f16>(product, product.integers(), cubes, 1);
        assert_merged(&got[0], product, &format!("{cubes} cubes"));
        assert!(
            arrivals.iter().all(|&a| a == 0),
            "{cubes} cubes: {arrivals:?}"
        );
    }
}

/// The counters a merge leaves at zero take the next launch as they took the first.
#[test]
fn the_last_arrival_merges_again_on_the_counters_it_left() {
    let client = cubecl::test_device().client();
    if !merges_on_arrival(&client) {
        return;
    }
    let product = Product::single(12, 16, 20);
    let (got, arrivals) = run_merged_stream::<half::f16>(product, product.integers(), 7, 3);
    assert_merged(&got[2], product, "a third launch");
    assert!(arrivals.iter().all(|&a| a == 0), "{arrivals:?}");
}

/// Sums whose rounding depends on the order of their parts come out the same bits every launch,
/// whichever cube arrives last: the parts are added in the order of their runs. The output is
/// `f32`, the sum's own element, since a cast to `f16` would round away the bits an order flips.
#[test]
fn the_last_arrival_sums_the_same_bits_every_launch() {
    let client = cubecl::test_device().client();
    if !merges_on_arrival(&client) {
        return;
    }
    let product = Product::single(12, 16, 20);
    let Product { m, n, k, .. } = product;
    let a = (0..m * k).map(|i| (i as f32 * 0.37).sin() * 3.1).collect();
    let b = (0..k * n).map(|i| (i as f32 * 0.71).cos() / 1.7).collect();
    let (got, _) = run_merged_stream::<f32>(product, (a, b), 29, 8);
    let bits = |out: &HostData| -> Vec<u32> {
        (0..m * n)
            .map(|i| out.get_f32(&[0, i / n, i % n]).to_bits())
            .collect()
    };
    let first = bits(&got[0]);
    for (launch, out) in got.iter().enumerate().skip(1) {
        assert_eq!(bits(out), first, "launch {launch} differs from the first");
    }
}
